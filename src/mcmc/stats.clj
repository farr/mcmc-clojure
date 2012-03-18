(ns mcmc.stats
  (:import (org.apache.commons.math.random RandomGenerator)))

(defn mean
  "Returns the mean of a sequence."
  [xs]
  (let [xs (double-array xs)]
    (/ (areduce xs i sum (double 0.0) (+ sum (aget xs i)))
       (alength xs))))

(defn variance
  "Returns the unbiased estimator for the variance of a sequence.
  Optional second argument is the pre-computed mean of the sequence."
  ([xs] (variance xs (mean xs)))
  ([xs mu]
     (let [xs (double-array xs)
           mu (double mu)]
       (/ (areduce xs i sum (double 0.0) (let [d (- (aget xs i) mu)] (+ sum (* d d))))
          (- (alength xs) 1)))))

(defn std
  "Returns the square root of the variance of a sequence (note that
  this is a biased estimator for the standard deviation).  Optional
  second argument is the mean of the sequence."
  ([xs] (std xs (mean xs)))
  ([xs mu] (Math/sqrt (double (variance xs mu)))))

;; The algorithm here is basically to generate homogeneous Poisson
;; process samples at the maximum rate of f, and then accept/reject
;; them according to the value of f at the sample time.
(defn inhomogeneous-poisson-array
  "Returns a double-array of sample times from the inhomogeneous
  poisson process defined by f.  The keyword arguments give the
  minimum and maximum times for the process and the maximum value of
  f, the rate.  The returned array will be in increasing-time order."
  ([f rng]
     (inhomogeneous-poisson-array f rng {}))
  ([f ^RandomGenerator rng {:keys [tmin tmax fmax] :or {tmin 0 tmax 1 fmax 1}}]
     (double-array
      (reverse
       (loop [tcur tmin samples ()]
         (if (> tcur tmax)
           samples
           (let [next-exp (- (Math/log (- 1.0 (.nextDouble rng))))
                 next-t (+ tcur (/ next-exp fmax))]
             (if (> next-t tmax)
               samples
               (if (<= (* fmax (.nextDouble rng)) (f next-t))
                 (recur next-t (cons next-t samples))
                 (recur next-t samples))))))))))

(defn p-value
  "Returns the p-value (percentile) of the given value in the given
  sequence."
  [x xs]
  (let [x (double x)]
    (loop [count (long 0)
           N (long 0)
           xs xs]
      (if (seq xs)
        (if (< (double (first xs)) x)
          (recur (+ count (long 1))
                 (+ N (long 1))
                 (rest xs))
          (recur count
                 (+ N (long 1))
                 (rest xs)))
        (/ (double count) (double N))))))

(defn- all-equal?
  "Returns true only if all the elements of coll compare equal."
  [coll compare]
  (if-let [coll (seq coll)]
    (loop [x (first coll)
           xs (rest coll)]
      (if-let [xs (seq xs)]
        (and (= (compare x (first xs)) 0)
             (recur x (rest xs)))
        true))
    true))

(defn- divide-by
  "Returns [satisfies-pred not-satisfies-pred] from the given
  collection and predicate."
  [pred? coll]
  (loop [satisfies '()
         not-satisfies '()
         coll coll]
    (if-let [coll (seq coll)]
      (if (pred? (first coll))
        (recur (conj satisfies (first coll))
               not-satisfies
               (rest coll))
        (recur satisfies
               (conj not-satisfies (first coll))
               (rest coll)))
      [satisfies not-satisfies])))

(defn- find-nth
  "Returns the nth element from a (hypothetical) sort of the given
  collection.  This can be performed in O(N) time, as opposed to the
  O(N*log(N)) sort."
  ([n coll]
     (find-nth n coll compare))
  ([n coll compare]
     (when (seq coll)
       (if (all-equal? coll compare)
         (first coll)
         (let [pivot (first coll)
               [lt gte] (divide-by #(< (compare % pivot) 0) (rest coll))
               n-lt (count lt)]
           (cond (< n n-lt) (recur n lt compare)
                 (= n n-lt) pivot
                 :else (recur (- n n-lt 1) gte compare)))))))

(defn percentile
  "Returns the element that is closest to the given percentile in the
  given collection.  Uses compare to sort the collection unless a
  specialized comparison function is given."
  ([p coll]
     (percentile p coll compare))
  ([p coll compare]
     (let [n (count coll)
           i (Math/round (double (* (- n 1) p)))]
       (find-nth i coll compare))))

(defn- sum-logs-to-n
  "Returns the sum of the numbers from 1 to n (inclusive)."
  [n]
  (loop [i (int n) lnf (double 0.0)]
    (if (<= i (int 1))
      lnf
      (recur (- i (int 1)) (+ lnf (Math/log (double i)))))))

(defn log-poisson-pdf
  "Returns the log of the value of the poisson PDF with rate lambda at
  n counts."
  [lam n]
  (let [lam (double lam)
        n (int n)
        log-nf (double (sum-logs-to-n n))]
    (- (* n (Math/log lam))
       (+ log-nf lam))))