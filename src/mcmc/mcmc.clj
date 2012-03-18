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