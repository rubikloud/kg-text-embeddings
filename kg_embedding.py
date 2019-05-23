import collections
import time
import sklearn.feature_extraction
import sklearn.preprocessing
import numpy as np 
from scipy import sparse 
import tensorflow as tf 


def init_nunif(sz, bnd=None):
    if bnd is None: 
        if len(sz) >= 2:
            bnd = np.sqrt(6) / np.sqrt(sz[0] + sz[1])
        else:
            bnd = 1.0 / np.sqrt(sz[0])
    return np.random.uniform(low=-bnd, high=bnd, size=sz)


def tf_cconv(a, b):
    a_fft = tf.fft(tf.complex(a, 0.0))
    b_fft = tf.fft(tf.complex(b, 0.0))
    ifft = tf.ifft(a_fft * b_fft)
    return tf.cast(tf.real(ifft), 'float32')

def np_cconv(a, b):
    a_fft = np.fft.fft(a)
    b_fft = np.fft.fft(b)
    return np.fft.ifft(a_fft * b_fft).real

def tf_ccorr(a, b):
    a_fft = tf.fft(tf.complex(a, 0.0))
    b_fft = tf.fft(tf.complex(b, 0.0))
    ifft = tf.ifft(tf.conj(a_fft) * b_fft)
    return tf.cast(tf.real(ifft), 'float32')

def np_ccorr(a, b):
    a_fft = np.fft.fft(a)
    b_fft = np.fft.fft(b)
    return np.fft.ifft(np.conj(a_fft) * b_fft).real

def vec_mat_mul(v, M):
    product = tf.matmul(tf.expand_dims(v,-2), M)
    return tf.squeeze(product, -2)


class BaseModel(object):
    def __init__(self, ne, nr, dim, samplef, **kwargs):
        self.samplef = samplef 
        self.ne = ne 
        self.nr = nr 
        self.dim = dim 
        self.pairwise = kwargs.pop("pairwise", True)  
        self.epochs = kwargs.pop("epochs",200)
        self.batch_size = kwargs.pop("batch_size",1024)
        self.learning_rate = kwargs.pop("learning_rate",0.01)
        self.reg = kwargs.pop("reg",0.0)
        self.margin = kwargs.pop("margin",1.0)

        self.param_names = [] # list of parameters names needed for computing the score of a triplet
        self.E_shape = [self.ne, self.dim]
        self.R_shape = [self.nr, self.dim]
        self.reg_loss = tf.constant(0, dtype=tf.float32) 


    def _on_epoch_begin(self):
        self.indices = np.arange(len(self.X))
        np.random.shuffle(self.indices)

    def _get_batch(self, idx):
        indices = self.indices[idx*self.batch_size : (idx+1)*self.batch_size]
        pos = self.X[indices]
        neg = np.array([self.samplef(fact, self.ne) for fact in pos])
        subjs = pos[:,0]
        objs = pos[:,2]
        preds = pos[:,1]
        neg_subjs = neg[:,0]
        neg_objs = neg[:,2]
        return {self.ps:subjs, self.po:objs, self.ns:neg_subjs, self.no:neg_objs, self.p:preds}

    def _add_param(self, name, shape, bnd=None):
        init_vals = init_nunif(shape, bnd)
        var = tf.Variable(init_vals, dtype=tf.float32, name=name) 
        setattr(self, name, var)
        self.param_names.append(name)
        x = tf.nn.l2_loss(var)
        self._regularize(var)


    def create_params(self):
        self._add_param("E", self.E_shape)
        self._add_param("R", self.R_shape)


    def gather(self, s, p, o):
        E_s = tf.gather(self.E, s)
        R = tf.gather(self.R, p)
        E_o = tf.gather(self.E, o)
        return E_s, R, E_o 

    def gather_np(self, si, pi, oi):
        es = self.E[si]
        eo = self.E[oi]
        r = self.R[pi]
        return es, r, eo 


    def train_score(self, s, p, o):
        raise NotImplementedError("train_score should be defined by the inheriting class")

    def _regularize(self, var):
        if self.reg > 0: 
            self.reg_loss += tf.nn.l2_loss(var)

    def train_loss(self, score_pos, score_neg):
        if self.pairwise:
            rank_loss = tf.reduce_sum(tf.maximum(0.0, self.margin - score_pos + score_neg))
        else:
            # Logistic loss 
            rank_loss = tf.reduce_sum(tf.nn.softplus(-score_pos) + tf.nn.softplus(score_neg))
        return rank_loss + self.reg * self.reg_loss 


    def fit(self, X, y=None):
        '''
        X : list/iterable of (subject, object, predicate) triplets
        y : ignored (assumes all examples are positive)
        '''
        self.X = np.array(X) 
        self.ps = tf.placeholder(tf.int32, [self.batch_size])
        self.p = tf.placeholder(tf.int32, [self.batch_size])
        self.po = tf.placeholder(tf.int32, [self.batch_size])
        self.ns = tf.placeholder(tf.int32, [self.batch_size])
        self.no = tf.placeholder(tf.int32, [self.batch_size])

        self.create_params()

        score_pos = self.train_score(self.ps, self.p, self.po)
        score_neg = self.train_score(self.ns, self.p, self.no)
        self.loss = self.train_loss(score_pos, score_neg)
        
        self._optimize() 


    def _run_epoch(self, optimizer):
        start = time.time() 
        self._on_epoch_begin()
        tot_loss = 0
        self.cur_epoch += 1

        for b in range(len(self.X)//self.batch_size):
            feed_dict = self._get_batch(b)
            _,l = self.sess.run([optimizer,self.loss], feed_dict=feed_dict)
            tot_loss += l 

        avg_loss = tot_loss / (len(self.X)//self.batch_size * self.batch_size)

        t = time.time() - start 
        print("Epoch: %i/%i; loss = %.9f (%.1f s)" %(self.cur_epoch+1,self.epochs,avg_loss,t), end="\r")
        if (self.cur_epoch+1)%10 == 0:
            print("")


    def _optimize(self):
        opt1 = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.cur_epoch = 0

        print("Starting training")
        for epoch in range(self.epochs):
            self._run_epoch(opt1) 
        print("")

        tf_objects = [getattr(self, attr) for attr in self.param_names]
        vals = self.sess.run(tf_objects)
        for attr,val in zip(self.param_names,vals):
            setattr(self, attr, val)


class TransE(BaseModel):
    def __init__(self, ne, nr, dim, samplef, **kwargs):
        BaseModel.__init__(self, ne, nr, dim, samplef, **kwargs)

    def train_score(self, s, p, o):
        E_s, R, E_o = self.gather(s, p, o)
        return -tf.reduce_sum(tf.abs(E_s + R - E_o), axis=-1)

    def score(self, si, pi, oi):
        es, r, eo = TransE.gather_np(self, si, pi, oi)
        return -np.sum(np.abs(es + r - eo), axis=-1)


class TransR(BaseModel):
    def __init__(self, ne, nr, dim, samplef, **kwargs):
        BaseModel.__init__(self, ne, nr, dim, samplef, **kwargs)

    def create_params(self):
        BaseModel.create_params(self)
        self._add_param("Mr", [self.nr, self.dim, self.dim])

    def gather(self, s, p, o):
        E_s = tf.gather(self.E, s)
        R = tf.gather(self.R, p)
        Mr = tf.gather(self.Mr, p)
        E_o = tf.gather(self.E, o)
        return E_s, R, Mr, E_o 

    def gather_np(self, si, pi, oi):
        es, r, eo = BaseModel.gather_np(self, si, pi, oi)
        Mr = self.Mr[pi]
        return es, r, Mr, eo

    def train_score(self, s, p, o):
        E_s, R, Mr, E_o = self.gather(s, p, o)
        # Mr : [batch_size x dim x dim]
        # E_s, E_o : [batch_size x dim]
        E_sr = vec_mat_mul(E_s, Mr)
        E_or = vec_mat_mul(E_o, Mr)
        return -tf.reduce_sum(tf.abs(E_sr + R - E_or), axis=-1)

    def score(self, si, pi, oi):
        es, r, Mr, eo = TransR.gather_np(self, si, pi, oi)
        esp = np.matmul(es, Mr)
        eop = np.matmul(eo, Mr)
        return -np.sum(np.abs(esp + r - eop), axis=-1)


class RESCAL(BaseModel):
    def __init__(self, ne, nr, dim, samplef, **kwargs):
        BaseModel.__init__(self, ne, nr, dim, samplef, **kwargs)
        self.R_shape = [self.nr, self.dim, self.dim]
        self.learning_rate = kwargs.pop("learning_rate",0.1)

    def train_score(self, s, p, o):
        E_s, R, E_o = self.gather(s, p, o)
        return tf.reduce_sum(vec_mat_mul(E_s, R) * E_o, axis=-1)

    def score(self, si, pi, oi):
        es, r, eo = RESCAL.gather_np(self, si, pi, oi)
        return np.sum(np.matmul(es, r) * eo, axis=-1)


class HolE(BaseModel):
    def __init__(self, ne, nr, dim, samplef, **kwargs):
        BaseModel.__init__(self, ne, nr, dim, samplef, **kwargs)
        self.learning_rate = kwargs.pop("learning_rate",0.1)
        
    def train_score(self, s, p, o):
        E_s, R, E_o = self.gather(s, p, o)
        return tf.reduce_sum(R * tf_ccorr(E_s, E_o), axis=-1)

    def score(self, si, pi, oi):
        es, r, eo = HolE.gather_np(self, si, pi, oi)
        return np.sum(r * np_ccorr(es, eo), axis=-1)


class SE(BaseModel):
    def __init__(self, ne, nr, dim, samplef, **kwargs):
        BaseModel.__init__(self, ne, nr, dim, samplef, **kwargs)
        self.R_shape = [self.nr, self.dim, self.dim]

    def create_params(self):
        BaseModel.create_params(self)
        self.param_names.remove("R")
        self._add_param("R1", self.R_shape)
        self._add_param("R2", self.R_shape)

    def gather(self, s, p, o):
        E_s = tf.gather(self.E, s)
        R1 = tf.gather(self.R1, p)
        R2 = tf.gather(self.R2, p)
        E_o = tf.gather(self.E, o)
        return E_s, E_o, R1, R2

    def gather_np(self, si, pi, oi):
        es, r, eo = BaseModel.gather_np(self, si, pi, oi)
        R1 = self.R1[pi]
        R2 = self.R2[pi]
        return es, eo, R1, R2 

    def train_score(self, s, p, o):
        E_s, E_o, R1, R2 = self.gather(s, p, o)
        # E_s, E_o : [batch_size x dim]
        # R1, R2 : [batch_size x dim x dim]
        E_sr = vec_mat_mul(E_s, R1)
        E_or = vec_mat_mul(E_o, R2)
        return -tf.reduce_sum(tf.abs(E_sr - E_or), axis=-1)

    def score(self, si, pi, oi):
        es, eo, R1, R2 = SE.gather_np(self, si, pi, oi)
        esr = np.matmul(es, R1)
        eor = np.matmul(eo, R2)
        return -np.sum(np.abs(esr - eor), axis=-1)


class DistMult(BaseModel):
    def __init__(self, ne, nr, dim, samplef, **kwargs):
        BaseModel.__init__(self, ne, nr, dim, samplef, **kwargs)
        self.learning_rate = kwargs.pop("learning_rate",0.1)

    def train_score(self, s, p, o):
        E_s, R, E_o = self.gather(s, p, o)
        return tf.reduce_sum(E_s * R * E_o, axis=-1)

    def score(self, si, pi, oi):
        es, r, eo = DistMult.gather_np(self, si, pi, oi)
        return np.sum(es * r * eo, axis=-1)


class BaseWordVectorsModel(BaseModel):
    def __init__(self, ne, nr, dim, samplef, word_ids, word_init=None, weighted=True, pe=False, 
        tfidf_weights=False, **kwargs):
        '''
        word_ids:   length ne list/iterable, where word_ids[i] is a list/iterable of indices 
                    of words associated with entity i 
        word_init:  length nw list/iterable, where word_init[i] is a numpy array indicating the 
                    initial value to assign to the vector for word i, or None if no intial
                    value is avaiable. 
        '''
        BaseModel.__init__(self, ne, nr, dim, samplef, **kwargs)
        self.word_ids = word_ids 
        assert len(self.word_ids) == ne
        self.nw = max(max(ids) for ids in self.word_ids if len(ids) > 0) + 1
        self.word_init = word_init 
        if self.word_init is not None:
            text_dim = next(len(x) for x in self.word_init if x is not None)
            if text_dim != self.dim:
                print("WARNING: Detected dimension %i word initialization vectors. Overriding dimension setting of %i." 
                            %(text_dim, self.dim))
                self.dim = text_dim 
        
        self.weighted = weighted
        self.pe = pe           
        self.tfidf_weights = tfidf_weights
        self.learning_rate = kwargs.pop("learning_rate",0.1)


    def create_params(self):
        # Since tensorflow does not support gathering of sparse matrices (needed for gathering a batch), 
        # we use the following workaround.
        # Store the sparse matrix as a list of sparse vectors (note the matrix is constant). 
        # Leave the sparse matrix A as a sparse placeholder.
        # Then to compute the entity embeddings of a batch, gather only the indices and values of A for that 
        # batch and feed it into the placeholder for A. Then compute E as usual. 

        # Build word matrix as in BaseModel
        nw = self.nw 
        assert all(max(ids) < self.nw for ids in self.word_ids if len(ids) > 0)
        # Assign unique dummy ids to entities with no words 
        for i in range(self.ne):
            if len(self.word_ids[i]) == 0:
                self.word_ids[i] = [self.nw]
                self.nw += 1

        self.nw += sum([len(words) == 0 for words in self.word_ids])
        W_init = init_nunif([self.nw, self.dim])
        if self.word_init is not None:
            for i in range(nw):
                if self.word_init[i] is not None:
                    W_init[i] = self.word_init[i]

        self.W = tf.Variable(W_init, dtype=tf.float32, name="W")
        self.param_names.append("W")
        self._regularize(self.W)

        if self.pe:
            # parameter-efficient weighting scheme 
            self.P = tf.Variable(np.random.uniform(low=-0.1, high=0.1, size=[self.nr,self.dim]), dtype=tf.float32, name="P")
            weights = tf.matmul(self.P, tf.transpose(self.W))
            self.B = tf.exp(weights)
            self.param_names.append("P")
            self.phase2_vars = [self.P]
        else:
            self.B = tf.Variable(np.ones([self.nr,self.nw]), dtype=tf.float32, trainable=self.weighted, name="B") 
            self.phase2_vars = [self.B]

        self.param_names.append("B")

        if self.tfidf_weights:  
            # Extract tf-idf weights 
            corpus = [" ".join(list(map(str,ids))) for ids in self.word_ids]
            vocab = map(str, range(self.nw))
            vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(vocabulary=vocab)
            self.A = vectorizer.fit_transform(corpus)                    
            self.A = sklearn.preprocessing.normalize(self.A, norm="l1", axis=1)

        else: 
            # Weights are just frequencies of words 
            # Create sparse matrix as a list of lists of indices and values 
            values = []
            row_ind = []
            col_ind = []
            no_word_id = nw 

            for i in range(self.ne):
                vals = collections.defaultdict(float)
                for j in self.word_ids[i]:
                    vals[j] += 1.0
                for j in vals.keys():
                    row_ind.append(i)
                    col_ind.append(j)
                    values.append(vals[j])

            self.A = sparse.csr_matrix((values, (row_ind, col_ind)), shape=[self.ne,self.nw], dtype=np.float32)
            self.A = sklearn.preprocessing.normalize(self.A, norm="l1", axis=1)
            
        self._add_param("R", self.R_shape)


    def _gather_A(self, idx):
        A_batch = sparse.coo_matrix(self.A[idx], dtype=np.float32)
        ret = tf.SparseTensorValue(indices=np.array([A_batch.row,A_batch.col]).T, values=A_batch.data, 
            dense_shape=[self.batch_size,self.nw])
        return ret 

    def _get_batch(self, idx):
        # Add sparse tensor values for A matrices to feed dict 
        feed_dict = BaseModel._get_batch(self, idx)
        feed_dict[self.A_ps] = self._gather_A(feed_dict[self.ps])
        feed_dict[self.A_po] = self._gather_A(feed_dict[self.po])
        feed_dict[self.A_ns] = self._gather_A(feed_dict[self.ns])
        feed_dict[self.A_no] = self._gather_A(feed_dict[self.no])
        return feed_dict 

    def gather(self, s, p, o):
        B = tf.gather(self.B, p)
        s_weights = self.A_s * B 
        s_weights /= tf.sparse_reduce_sum(s_weights, axis=1, keep_dims=True)
        o_weights = self.A_o * B 
        o_weights /= tf.sparse_reduce_sum(o_weights, axis=1, keep_dims=True)
        E_s = tf.sparse_tensor_dense_matmul(s_weights, self.W)
        E_o = tf.sparse_tensor_dense_matmul(o_weights, self.W)
        R = tf.gather(self.R, p)
        return E_s, R, E_o 


    def gather_np(self, si, pi, oi):
        B = self.B[pi]
        A_s = self.A[si] 
        A_o = self.A[oi] 
        s_weights = A_s.multiply(B)
        s_weights /= s_weights.sum(axis=-1)
        o_weights = A_o.multiply(B)
        o_weights /= o_weights.sum(axis=-1)
        es = np.array(sparse.csr_matrix.dot(s_weights, self.W))
        eo = np.array(sparse.csr_matrix.dot(o_weights, self.W))
        r = self.R[pi]
        return es, r, eo 


    def fit(self, X, y=None):
        '''
        X : list/iterable of (subject, object, predicate) triplets
        y : ignored (assumes all examples are positive)
        '''
        self.X = np.array(X) 
        self.ps = tf.placeholder(tf.int32, [self.batch_size])
        self.p = tf.placeholder(tf.int32, [self.batch_size])
        self.po = tf.placeholder(tf.int32, [self.batch_size])
        self.ns = tf.placeholder(tf.int32, [self.batch_size])
        self.no = tf.placeholder(tf.int32, [self.batch_size])

        self.create_params()

        # Note: this must be done after create_params because create_params changes nw 
        self.A_ps = tf.sparse_placeholder(dtype=tf.float32)
        self.A_po = tf.sparse_placeholder(dtype=tf.float32)
        self.A_ns = tf.sparse_placeholder(dtype=tf.float32)
        self.A_no = tf.sparse_placeholder(dtype=tf.float32)

        # Workaround so we don't have to rewrite all the train_score functions. Instead 
        # of passing in A_ps and A_po, store it in self before calling the function. 
        self.A_s, self.A_o = self.A_ps, self.A_po 
        score_pos = self.train_score(self.ps, self.p, self.po)
        self.A_s, self.A_o = self.A_ns, self.A_no 
        score_neg = self.train_score(self.ns, self.p, self.no)
        self.loss = self.train_loss(score_pos, score_neg)
        
        self._optimize() 


    def _optimize(self):
        if not self.weighted:
            BaseModel._optimize(self)

        else:
            phase1_vars = tf.trainable_variables() 
            for var in self.phase2_vars:
                phase1_vars.remove(var)
            opt1 = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, var_list=phase1_vars)
            opt2 = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss)

            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            self.sess.run(init)
            self.cur_epoch = 0

            print("Optimizing - phase 1")
            for epoch in range(self.epochs//4):
                self._run_epoch(opt1)
            print("")

            print("Optimizing - phase 2")
            for epoch in range(3*self.epochs//4):
                self._run_epoch(opt2)
            print("")

            tf_objects = [getattr(self, attr) for attr in self.param_names]
            vals = self.sess.run(tf_objects)
            for attr,val in zip(self.param_names,vals):
                setattr(self, attr, val)

        self.param_names.append("A")


class TransEWordVectors(BaseWordVectorsModel):
    def __init__(self, ne, nr, dim, samplef, word_ids, word_init=None, weighted=True, 
        pe=False, tfidf_weights=False, **kwargs):
        BaseWordVectorsModel.__init__(self, ne, nr, dim, samplef, word_ids, word_init=word_init, 
        weighted=weighted, pe=pe, tfidf_weights=tfidf_weights, **kwargs)

    def train_score(self, s, p, o):
        return TransE.train_score(self, s, p, o)

    def score(self, si, pi, oi):
        es, r, eo = TransEWordVectors.gather_np(self, si, pi, oi)
        return -np.sum(np.abs(es + r - eo), axis=-1, keepdims=False)


class TransRWordVectors(BaseWordVectorsModel):
    def __init__(self, ne, nr, dim, samplef, word_ids, word_init=None, weighted=True, 
        pe=False, tfidf_weights=False, **kwargs):
        BaseWordVectorsModel.__init__(self, ne, nr, dim, samplef, word_ids, word_init=word_init,
        weighted=weighted, pe=pe, tfidf_weights=tfidf_weights, **kwargs)

    def create_params(self):
        BaseWordVectorsModel.create_params(self)
        self._add_param("Mr", [self.nr, self.dim, self.dim])

    def gather(self, s, p, o):
        E_s, R, E_o = BaseWordVectorsModel.gather(self, s, p, o)
        Mr = tf.gather(self.Mr, p)
        return E_s, R, Mr, E_o 

    def gather_np(self, si, pi, oi):
        es, r, eo = BaseWordVectorsModel.gather_np(self, si, pi, oi)
        Mr = self.Mr[pi]
        return es, r, Mr, eo

    def train_score(self, s, p, o):
        return TransR.train_score(self, s, p, o)

    def score(self, si, pi, oi):
        es, r, Mr, eo = TransRWordVectors.gather_np(self, si, pi, oi)
        esp = np.matmul(es, Mr)
        eop = np.matmul(eo, Mr)
        return -np.sum(np.abs(esp + r - eop), axis=-1)


class RESCALWordVectors(BaseWordVectorsModel):
    def __init__(self, ne, nr, dim, samplef, word_ids, word_init=None, weighted=True,
        pe=False, tfidf_weights=False, **kwargs):
        BaseWordVectorsModel.__init__(self, ne, nr, dim, samplef, word_ids, word_init=word_init,
        weighted=weighted, pe=pe, tfidf_weights=tfidf_weights, **kwargs)
        self.R_shape = [self.nr, self.dim, self.dim]

    def train_score(self, s, p, o):
        return RESCAL.train_score(self, s, p, o)

    def score(self, si, pi, oi):
        es, r, eo = RESCALWordVectors.gather_np(self, si, pi, oi)
        return np.sum(np.matmul(es, r) * eo, axis=-1)


class HolEWordVectors(BaseWordVectorsModel):
    def __init__(self, ne, nr, dim, samplef, word_ids, word_init=None, weighted=True, 
        pe=False, tfidf_weights=False, **kwargs):
        BaseWordVectorsModel.__init__(self, ne, nr, dim, samplef, word_ids, word_init=word_init, 
        weighted=weighted, pe=pe, tfidf_weights=tfidf_weights, **kwargs)
        
    def train_score(self, s, p, o):
        return HolE.train_score(self, s, p, o)

    def score(self, si, pi, oi):
        es, r, eo = HolEWordVectors.gather_np(self, si, pi, oi)
        return np.sum(r * np_ccorr(es, eo), axis=-1)


class SEWordVectors(BaseWordVectorsModel):
    def __init__(self, ne, nr, dim, samplef, word_ids, word_init=None, weighted=True, 
        pe=False, tfidf_weights=False, **kwargs):
        BaseWordVectorsModel.__init__(self, ne, nr, dim, samplef, word_ids, word_init=word_init, 
        weighted=weighted, pe=pe, tfidf_weights=tfidf_weights, **kwargs)
        self.R_shape = [self.nr, self.dim, self.dim]

    def create_params(self):
        BaseWordVectorsModel.create_params(self)
        self._add_param("R1", self.R_shape)
        self._add_param("R2", self.R_shape)

    def gather(self, s, p, o):
        E_s, R, E_o = BaseWordVectorsModel.gather(self, s, p, o)
        R1 = tf.gather(self.R1, p)
        R2 = tf.gather(self.R2, p)
        return E_s, E_o, R1, R2

    def gather_np(self, si, pi, oi):
        es, r, eo = BaseWordVectorsModel.gather_np(self, si, pi, oi)
        R1 = self.R1[pi]
        R2 = self.R2[pi]
        return es, eo, R1, R2         

    def train_score(self, s, p, o):
        return SE.train_score(self, s, p, o)

    def score(self, si, pi, oi):
        es, eo, R1, R2 = SEWordVectors.gather_np(self, si, pi, oi)
        esr = np.matmul(es, R1)
        eor = np.matmul(eo, R2)
        return -np.sum(np.abs(esr - eor), axis=-1)


class DistMultWordVectors(BaseWordVectorsModel):
    def __init__(self, ne, nr, dim, samplef, word_ids, word_init=None, weighted=True, 
        pe=False, tfidf_weights=False, **kwargs):
        BaseWordVectorsModel.__init__(self, ne, nr, dim, samplef, word_ids, word_init=word_init, 
        weighted=weighted, pe=pe, tfidf_weights=tfidf_weights, **kwargs)

    def train_score(self, s, p, o):
        return DistMult.train_score(self, s, p, o)

    def score(self, si, pi, oi):
        es, r, eo = DistMultWordVectors.gather_np(self, si, pi, oi)
        return np.sum(es * r * eo, axis=-1)


class BaseFeatureSumModel(BaseModel):
    def __init__(self, ne, nr, dim, samplef, W_text, **kwargs):
        BaseModel.__init__(self, ne, nr, dim, samplef, **kwargs)
        self.orig_samplef = samplef 
        self.train_words = kwargs.pop("train_words",True)
        self.reg = 0.0        
        self.W_text = W_text 
        self.text_dim = next(len(x) for x in W_text if x is not None)
        self.R_shape = [self.nr, self.text_dim + self.dim]
                

    def _permute_indices(self, X):
        '''
        Permute entity indices such that all entities with unknown text embeddings (W_text[i] is None) 
        occur at the end. 
        '''
        # idx_map : old index to new index
        # inv_idx_map : new index to old index
        self.idx_map = [0] * self.ne 
        self.inv_idx_map = [0] * self.ne 
        self.num_unknown = sum(1 for x in self.W_text if x is None)
        num_known = sum(1 for x in self.W_text if x is not None)

        cnt_known = 0 
        cnt_unknown = 0 
        for i in range(self.ne):
            if self.W_text[i] is None:
                self.idx_map[i] = num_known + cnt_unknown
                cnt_unknown += 1
            else:
                self.idx_map[i] = cnt_known 
                cnt_known += 1
            self.inv_idx_map[self.idx_map[i]] = i 
        assert list(sorted(self.idx_map)) == list(range(self.ne))
        assert list(sorted(self.inv_idx_map)) == list(range(self.ne))

        # Update indices in training data x 
        for i in range(len(X)):
            X[i] = list(X[i])
            X[i][0] = self.idx_map[X[i][0]]
            X[i][2] = self.idx_map[X[i][2]]
        self.W_text = np.array([x for x in self.W_text if x is not None])

        # IMPORTANT: Update negative sampling function to undo the permutation because 
        # it references the internal KnowledgeGraph representation which is not permuted.
        # Then redo the permutation on the result. 
        def new_samplef(x, ne):
            res = self.orig_samplef((self.inv_idx_map[x[0]], x[1], self.inv_idx_map[x[2]]), ne)
            return (self.idx_map[res[0]], res[1], self.idx_map[res[2]])
        self.samplef = new_samplef 

        assert len(self.W_text) + self.num_unknown == self.ne 

        return np.array(X) 



    def _unpermute_indices(self):
        '''
        Undo the index permutation by rearranging the rows of self.E and self.W.
        '''
        for attr in ["E","W"]:
            arr = getattr(self,attr)
            new_arr = []
            for i in range(self.ne):
                # i is old index, idx_map[i] is new index
                new_arr.append(arr[self.idx_map[i]])
            setattr(self, attr, np.array(new_arr))


    def create_params(self):
        # compute average L1 norm of text vector
        norms = [np.mean(np.abs(vec)) for vec in self.W_text if vec is not None]
        avg_norm = np.mean(norms)

        self._add_param("E", self.E_shape)
        self._add_param("R", self.R_shape)

        if self.num_unknown > 0:
            print("%i entities with unknown text embeddings" %(self.num_unknown))
            W_known = tf.Variable(self.W_text, dtype=tf.float32, trainable=self.train_words)
            W_unknown = tf.Variable(init_nunif([self.num_unknown, self.text_dim]), dtype=tf.float32)
            self.W = tf.concat([W_known,W_unknown], axis=0, name="W")
            self.phase2_vars = [W_unknown]
            if self.train_words:
                self.phase2_vars.append(W_known)
        else:
            self.W = tf.Variable(self.W_text, dtype=tf.float32, trainable=self.train_words, name="W")            
            self.phase2_vars = [self.W]

        self.param_names.append("W")
        self._regularize(self.W)

        self.M = tf.Variable(np.zeros([self.text_dim,self.dim]), dtype=tf.float32, name="M")
        self.phase2_vars.append(self.M)
        self.param_names.append("M")

        # Replace each word vector w in self.W with w x self.M so we don't have to recompute this 
        # multiple times when doing prediction
        self.W = tf.matmul(self.W, self.M)


    def gather(self, s, p, o):
        E_s, R, E_o = BaseModel.gather(self, s, p, o)
        W_s = tf.gather(self.W, s)
        W_o = tf.gather(self.W, o)
        return E_s, W_s, R, E_o, W_o 


    def fit(self, X, y=None):
        X = self._permute_indices(X) 
        BaseModel.fit(self, X, y)
        self._unpermute_indices() 


    def _optimize(self):        
        phase1_vars = tf.trainable_variables() 
        if hasattr(self,"phase2_vars"):
            for var in self.phase2_vars:
                phase1_vars.remove(var) 
        opt1 = tf.train.AdagradOptimizer(self.learning_rate).minimize(self.loss, var_list=phase1_vars)
        opt2 = tf.train.AdagradOptimizer(0.01).minimize(self.loss)

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.cur_epoch = 0

        print("Optimizing - phase 1")
        for epoch in range(self.epochs//2):
            self._run_epoch(opt1)
        print("")

        print("Optimizing - phase 2")
        for epoch in range(self.epochs//2):
            self._run_epoch(opt2)
        print("")

        tf_objects = [getattr(self, attr) for attr in self.param_names]
        vals = self.sess.run(tf_objects)
        for attr,val in zip(self.param_names,vals):
            setattr(self, attr, val)


class TransEFeatureSum(BaseFeatureSumModel):
    def __init__(self, ne, nr, dim, samplef, W_text, **kwargs):
        BaseFeatureSumModel.__init__(self, ne, nr, dim, samplef, W_text, **kwargs)
        self.R_shape = [self.nr, self.dim]
        self.train_words = True

    def train_score(self, s, p, o):
        E_s, W_s, R, E_o, W_o = self.gather(s, p, o)
        E_s = E_s + W_s 
        E_o = E_o + W_o 
        return -tf.reduce_sum(tf.abs(E_s + R - E_o), axis=-1)

    def score(self, si, pi, oi):
        es = self.E[si] + self.W[si] 
        eo = self.E[oi] + self.W[oi] 
        r = self.R[pi]
        return -np.sum(np.abs(es + r - eo), axis=-1) 


class RESCALFeatureSum(BaseFeatureSumModel):
    def __init__(self, ne, nr, dim, samplef, W_text, **kwargs):
        BaseFeatureSumModel.__init__(self, ne, nr, dim, samplef, W_text, **kwargs)
        self.R_shape = [self.nr, self.dim, self.dim]
        self.learning_rate = kwargs.pop("learning_rate",0.1)
        self.reg = 0.01
        self.train_words = True

    def train_score(self, s, p, o):
        E_s, W_s, R, E_o, W_o = self.gather(s, p, o)
        E_s = E_s + W_s 
        E_o = E_o + W_o 
        return tf.reduce_sum(vec_mat_mul(E_s, R) * E_o, axis=-1)

    def score(self, si, pi, oi):
        es = self.E[si] + self.W[si] 
        eo = self.E[oi] + self.W[oi] 
        r = self.R[pi]
        return np.sum(np.matmul(es, r) * eo, axis=-1)   


class HolEFeatureSum(BaseFeatureSumModel):
    def __init__(self, ne, nr, dim, samplef, W_text, **kwargs):
        BaseFeatureSumModel.__init__(self, ne, nr, dim, samplef, W_text, **kwargs)
        self.R_shape = [self.nr, self.dim]
        self.learning_rate = kwargs.pop("learning_rate",0.1)
        self.reg = 0.01
        self.train_words = True

    def train_score(self, s, p, o):
        E_s, W_s, R, E_o, W_o = self.gather(s, p, o)
        E_s = E_s + W_s 
        E_o = E_o + W_o 
        return tf.reduce_sum(R * tf_ccorr(E_s, E_o), axis=-1)

    def score(self, si, pi, oi):
        es = self.E[si] + self.W[si] 
        eo = self.E[oi] + self.W[oi] 
        r = self.R[pi]
        return np.sum(r * np_ccorr(es, eo), axis=-1)


class TransRFeatureSum(BaseFeatureSumModel): 
    def __init__(self, ne, nr, dim, samplef, W_text, **kwargs):
        BaseFeatureSumModel.__init__(self, ne, nr, dim, samplef, W_text, **kwargs)
        self.R_shape = [self.nr, self.dim]
        self.reg = 0.01
        self.train_words = False
        
    def create_params(self):
        BaseFeatureSumModel.create_params(self)
        self._add_param("Mr", self.R_shape + [self.R_shape[-1]])
        
    def gather(self, s, p, o):
        E_s, W_s, R, E_o, W_o = BaseFeatureSumModel.gather(self, s, p, o)
        Mr = tf.gather(self.Mr, p)
        return E_s, W_s, R, Mr, E_o, W_o 

    def train_score(self, s, p, o):
        E_s, W_s, R, Mr, E_o, W_o = self.gather(s, p, o)
        E_s = E_s + W_s 
        E_o = E_o + W_o 
        E_sr = vec_mat_mul(E_s, Mr)
        E_or = vec_mat_mul(E_o, Mr)
        return -tf.reduce_sum(tf.abs(E_sr + R - E_or), axis=-1)
        
    def score(self, si, pi, oi):
        es = self.E[si] + self.W[si] 
        eo = self.E[oi] + self.W[oi] 
        r = self.R[pi]
        Mr = self.Mr[pi]
        esp = np.matmul(es, Mr)
        eop = np.matmul(eo, Mr)
        return -np.sum(np.abs(esp + r - eop), axis=-1)


class SEFeatureSum(BaseFeatureSumModel):
    def __init__(self, ne, nr, dim, samplef, W_text, **kwargs):
        BaseFeatureSumModel.__init__(self, ne, nr, dim, samplef, W_text, **kwargs)
        self.R_shape = [self.nr, self.dim, self.dim]
        self.reg = 0.01      
        self.train_words = False
        
    def create_params(self):
        BaseFeatureSumModel.create_params(self)
        self._add_param("R1", self.R_shape)
        self._add_param("R2", self.R_shape)

    def gather(self, s, p, o):
        E_s, W_s, R, E_o, W_o = BaseFeatureSumModel.gather(self, s, p, o)
        R1 = tf.gather(self.R1, p)
        R2 = tf.gather(self.R2, p)
        return E_s, W_s, E_o, W_o, R1, R2 

    def train_score(self, s, p, o):
        E_s, W_s, E_o, W_o, R1, R2  = self.gather(s, p, o)
        E_s = E_s + W_s 
        E_o = E_o + W_o 
        E_sr = vec_mat_mul(E_s, R1)
        E_or = vec_mat_mul(E_o, R2)
        return -tf.reduce_sum(tf.abs(E_sr - E_or), axis=-1)

    def score(self, si, pi, oi):
        es = self.E[si] + self.W[si] 
        eo = self.E[oi] + self.W[oi] 
        R1 = self.R1[pi]
        R2 = self.R2[pi]
        esr = np.matmul(es, R1)
        eor = np.matmul(eo, R2)
        return -np.sum(np.abs(esr - eor), axis=-1)

        
class DistMultFeatureSum(BaseFeatureSumModel):
    def __init__(self, ne, nr, dim, samplef, W_text, **kwargs):
        BaseFeatureSumModel.__init__(self, ne, nr, dim, samplef, W_text, **kwargs)
        self.R_shape = [self.nr, self.dim] 
        self.train_words = True

    def train_score(self, s, p, o):
        E_s, W_s, R, E_o, W_o = self.gather(s, p, o)
        E_s = E_s + W_s 
        E_o = E_o + W_o 
        return tf.reduce_sum(E_s * R * E_o, axis=-1)

    def score(self, si, pi, oi):
        es = self.E[si] + self.W[si] 
        eo = self.E[oi] + self.W[oi] 
        r = self.R[pi]
        return np.sum(es * r * eo, axis=-1)

