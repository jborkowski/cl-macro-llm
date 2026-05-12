;;; tests.lisp — generated from j14i/cl-ds row 170
;;; instruction: Write a Common Lisp macro `check-same-length` that evaluates two sequence expressions exactly once and signals an error 
;;; category: validation | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((check-same-length keys values) . (LET ((#:G1 KEYS) (#:G2 VALUES)) (UNLESS (= (LENGTH #:G1) (LENGTH #:G2)) (ERROR "Sequences have different lengths: ~A vs ~A" (LENGTH #:G1) (LENGTH #:G2))))))
