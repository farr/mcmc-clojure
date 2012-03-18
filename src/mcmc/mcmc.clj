(ns mcmc.mcmc
  (:import (org.apache.commons.math.random RandomGenerator Well44497b)))

(defn make-mcmc-sampler
  "Returns a function that produces a sequence of MCMC samples from a
  given initial sample.  Arguments are as follows:

log-likelihood : state -> double : returns the log of the likelihood
of a given state.

log-prior : state -> double : returns the log of the prior of a given
state.

jump : either a single jump propoosal (state -> state) or a sequence
of jump proposals.  If a sequence, the jumps will cycle through the
sequence.   

log-jump-prob : either a single function (state -> state ->
 [log-forward-prob log-backward-prob]) or a sequence of such functions
 corresponding to the forward and backward jump probability densities
 of the sequence of jump proposals.

rng : an optional RandomGenerator object from the Apache commons math
library (if none is given, a fresh Well44497b RNG will be allocated
with the default seeding procedure)."
  ([log-likelihood log-prior jump log-jump-prob]
     (make-mcmc-sampler log-likelihood log-prior jump log-jump-prob (Well44497b.)))
  ([log-likelihood log-prior jump log-jump-prob ^RandomGenerator rng]
     (let [jump (seq (if (coll? jump) jump [jump]))
           log-jump-prob (seq (if (coll? log-jump-prob) log-jump-prob [log-jump-prob]))]
       (let [next-mcmc (fn [state ll lp jumps log-jump-probs]
                         (let [jumps (or (seq jumps) jump)
                               log-jump-probs (or (seq log-jump-probs) log-jump-prob)
                               jump (first jumps)
                               log-jump-prob (first log-jump-probs)
                               ll (double ll)
                               lp (double lp)
                               new-state (jump state)
                               new-ll (double (log-likelihood new-state))
                               new-lp (double (log-prior new-state))
                               [log-forward log-backward] (log-jump-prob state new-state)
                               log-forward (double log-forward)
                               log-backward (double log-backward)]
                           (let [paccept (- (+ new-ll new-lp log-backward)
                                            (+ ll lp log-forward))]
                             (if (or (> paccept 1.0)
                                     (< (Math/log (.nextDouble rng)) paccept))
                               [new-state new-ll new-lp (rest jumps) (rest log-jump-probs)]
                               [state ll lp (rest jumps) (rest log-jump-probs)]))))]
         (fn [init]
           (let [mcmc-seq (fn mcmc-seq [current ll lp jump log-jump-prob]
                            (let [[next ll lp jumps log-jump-probs] (next-mcmc current ll lp jump log-jump-prob)]
                              (lazy-seq
                               (cons current (mcmc-seq next ll lp jumps log-jump-probs)))))]
             (mcmc-seq init (log-likelihood init) (log-prior init) jump log-jump-prob)))))))

(defn make-affine-sampler
  "Returns an affine-invariant sampler, following Goodman, J. & Weare,
  J., 2010, Comm. App. Math. Comp. Sci., 5, 65."
  ([log-likelihood log-prior]
     (make-affine-sampler log-likelihood log-prior (Well44497b.)))
  ([log-likelihood log-prior ^RandomGenerator rng]
     (letfn [(samples-seq [samples lls lps rngs]
               (let [n (count samples)
                     n-split (int (/ n 2))
                     [samples1 samples2] (split-at n-split samples)
                     [lls1 lls2] (split-at n-split lls)
                     [lps1 lps2] (split-at n-split lps)
                     [rngs1 rngs2] (split-at n-split rngs)
                     samples1 (vec samples1)
                     samples2 (vec samples2)]
                 (assert (seq samples1) "Error: no samples in subset 1!")
                 (assert (seq samples2) "Error: no samples in subset 2!")
                 (letfn [(draw-new-state [state ll lp ^RandomGenerator rng draw-samples]
                           (let [u (.nextDouble rng)
                                 z (* (double 0.5)
                                      (+ (double 1.0)
                                         (+ (* (double 2.0) u)
                                            (* u u))))
                                 state (doubles state)
                                 other (doubles (get draw-samples (.nextInt rng (count draw-samples))))
                                 new-state (doubles
                                            (amap state i ret
                                                  (+ (aget other i)
                                                     (* z (- (aget state i) (aget other i))))))
                                 n (alength new-state)
                                 ll (double ll)
                                 lp (double lp)
                                 new-ll (double (log-likelihood new-state))
                                 new-lp (double (log-prior new-state))
                                 accept-p (- (+ new-ll new-lp)
                                             (+ ll lp))
                                 accept-p (+ accept-p (* (- n (int 1)) (Math/log z)))]
                             (if (< (Math/log (.nextDouble rng)) accept-p)
                               [new-state new-ll new-lp]
                               [state ll lp])))]
                   (let [new-states-ll-lp1 (pmap (fn [state ll lp rng]
                                                   (draw-new-state state ll lp rng samples2))
                                                 samples1 lls1 lps1 rngs1)
                         samples1 (vec (map #(get % 0) new-states-ll-lp1))
                         new-states-ll-lp2 (pmap (fn [state ll lp rng]
                                                   (draw-new-state state ll lp rng samples1))
                                                 samples2 lls2 lps2 rngs2)
                         samples (concat samples1 (map #(get % 0) new-states-ll-lp2))
                         lls (concat (map #(get % 1) new-states-ll-lp1)
                                     (map #(get % 1) new-states-ll-lp2))
                         lps (concat (map #(get % 2) new-states-ll-lp1)
                                     (map #(get % 2) new-states-ll-lp2))]
                     (lazy-seq (cons samples (samples-seq samples lls lps rngs)))))))]
       (fn [init-samples]
         (samples-seq init-samples
                      (map log-likelihood init-samples)
                      (map log-prior init-samples)
                      (map (fn [samp] (Well44497b. (.nextInt rng))) init-samples))))))