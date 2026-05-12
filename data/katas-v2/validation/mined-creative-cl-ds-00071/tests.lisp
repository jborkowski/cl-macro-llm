;;; tests.lisp — generated from j14i/cl-ds row 71
;;; instruction: Write a Common Lisp macro `check-range` that evaluates value, low, and high each exactly once and signals an error unles
;;; category: validation | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((check-range x 0 100) . (LET ((#:G1 X) (#:G2 0) (#:G3 100)) (UNLESS (<= #:G2 #:G1 #:G3) (ERROR "Value ~S out of range [~S, ~S]" #:G1 #:G2 #:G3)) #:G1)))
