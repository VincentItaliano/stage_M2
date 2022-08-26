import tensorflow as tf
import numpy as np

def create_bank_of_points(n,use_Halton=True):
    
    from scipy.stats import qmc

    if use_Halton:
        sampler = qmc.Halton(d=2,scramble=False) 
        bank =  sampler.random(n=n)
    else:
        bank =  np.random.uniform(0,1,[n, 2]).astype(np.float32)

    np.save("./bank.npy",bank)

class sampler:

    def __init__(self, lb, ub, nb_cell):

        self.lb = lb
        self.ub = ub

        self.nb_cell = nb_cell

        self.centers = self.make_centers()
        self.nb_centers=len(self.centers)
        
        self._bank = tf.constant(np.load("../data/bank.npy"),dtype=np.float32)
        scale = tf.stack([self.db,self.da])
        shift = tf.stack([-0.5*self.db, -0.5 * self.da])
        self._bank *= scale
        self._bank += shift
        
        self.mask = tf.ones([3, 3, 1, 1])
        
    def make_centers(self):

        self.lb_border = tf.linspace(self.lb[1],self.ub[1], self.nb_cell[0]+1)
        a = self.lb_border[:-1]
        self.da = a[1] - a[0]
        a = a + self.da/2

        self.ub_border = tf.linspace(self.lb[2],self.ub[2], self.nb_cell[1]+1)
        b = self.ub_border[:-1] 
        self.db = b[1]-b[0]
        b = b + self.db/2
        
        aa,bb = tf.meshgrid(a,b)
        aa = tf.reshape(aa,[-1])
        bb = tf.reshape(bb,[-1])
        aabb = tf.stack([bb,aa],axis=1)

        return aabb
    
    def sample(self, func, nb_sample):

        self.t_r = tf.random.uniform((self.nb_centers,1), self.lb[0], self.ub[0], dtype='float32')
        X = tf.concat([self.t_r, self.centers], axis=1)
        values_at_center = func(X)
        current=tf.constant(np.random.randint(0,10_000),dtype=tf.int32)
        return self.sample_acc(values_at_center, current, nb_sample)
    
    @tf.function
    def sample_acc(self, values_at_center, current, nb_sample):
        print("traçage de la fonction sample")
        
        # smoothing
        values_matrix = tf.reshape(values_at_center,[1,self.nb_cell[0],self.nb_cell[1],1])
        C = tf.nn.conv2d(values_matrix, self.mask, [1, 1, 1, 1], padding="SAME")
        values_at_center = tf.reshape(C[0,:,:,0],[-1])
        
        # normalize density
        values_at_center-=tf.reduce_min(values_at_center)
        values_at_center/=tf.reduce_sum(values_at_center)
        
        samples=[]
        
        for i in range(len(values_at_center)):
            center = self.centers[i]
            value = values_at_center[i]
            n = tf.floor(value*nb_sample) + 10
            n = tf.cast(n,tf.int32)

            sample = self._bank[current:(current + n)]
            current = current+n
            sample += center
            samples.append(sample)

        return tf.concat(samples,axis=0)[:nb_sample]
        