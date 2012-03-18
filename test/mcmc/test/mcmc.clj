(ns mcmc.test.mcmc
  (:use mcmc.mcmc mcmc.stats clojure.test)
  (:import (java.io FileWriter BufferedWriter)
           (org.apache.commons.math.random Well44497b)))

(def rng (Well44497b.))

(defn close?
  "Returns true only if the given numbers are within the given absolute and relative error."
  ([a b] (close? a b {}))
  ([a b {:keys [epsabs epsrel] :or {epsabs 1e-8 epsrel 1e-8}}]
     (let [a (double a)
           b (double b)
           dx (Math/abs (- a b))
           ave (* 0.5 (+ (Math/abs a) (Math/abs b)))]
       (<= dx (+ epsabs (* ave epsrel))))))

(deftest mcmc-gaussian-1D-test
  (let [^Well44497b rng rng
        mu 3.5
        sigma 1.0
        log-sqrt-gauss (fn [x]
                         (let [dx (/ (- x mu) sigma)]
                           (- (/ (* dx dx) 4.0))))
        jump (fn [x]
               (if (< (.nextDouble rng) 0.3)
                 (- x (* 2 (.nextDouble rng)))
                 (+ x (* 2 (.nextDouble rng)))))
        log-jump-prob (fn [x y]
                        (if (< y x)
                          [(Math/log 0.3) (Math/log 0.7)]
                          [(Math/log 0.7) (Math/log 0.3)]))
        sampler (make-mcmc-sampler log-sqrt-gauss log-sqrt-gauss jump log-jump-prob rng)
        samples (take 10000 (sampler mu))]
    (comment
      (with-open [out (BufferedWriter. (FileWriter. "/tmp/gaussian.dat"))]
        (binding [*out* out]
          (doseq [x samples]
            (println x)))))
    (is (close? (mean samples) mu {:epsabs (* 6.0 (/ sigma (Math/sqrt (count samples)))) :epsrel 0.0}))
    (is (close? (std samples) sigma {:epsabs (* 12.0 (/ sigma (Math/sqrt (count samples)))) :epsrel 0.0}))))

(deftest mcmc-jump-cycle-test
  (let [^Well44497b rng rng
        mu 3.5
        sigma 2.0
        log-likelihood (fn [x] (let [dx (/ (- x mu) sigma)] (- (/ (* dx dx) 2.0))))
        jump1 (fn [x] (let [dx (- (.nextDouble rng) 0.5)] (+ x dx)))
        prob1 (fn [x y] [0.0 0.0])
        jump2 (fn [x] (let [dx (.nextGaussian rng)] (+ mu dx)))
        prop-log-gaussian (fn [x] (let [dx (- mu x)] (- (/ (* dx dx) 2.0))))
        prob2 (fn [x y] [(prop-log-gaussian y) (prop-log-gaussian x)])
        jumps [jump1 jump2]
        log-jump-probs [prob1 prob2]
        sampler (make-mcmc-sampler log-likelihood (fn [x] 0.0) jumps log-jump-probs rng)
        samples (take 10000 (take-nth 100 (sampler mu)))]
    (comment
      (with-open [out (BufferedWriter. (FileWriter. "/tmp/gaussian.dat"))]
        (binding [*out* out]
          (doseq [x samples]
            (println x)))))
    (is (close? (mean samples) mu {:epsabs (* 6.0 (/ sigma (Math/sqrt (count samples)))) :epsrel 0.0}))
    (is (close? (std samples) sigma {:epsabs (* 12.0 (/ sigma (Math/sqrt (count samples)))) :epsrel 0.0}))))