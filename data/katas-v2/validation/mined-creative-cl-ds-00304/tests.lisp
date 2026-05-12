;;; tests.lisp — generated from j14i/cl-ds row 304
;;; instruction: Write a Common Lisp macro `assert-in-range` that evaluates value, low, and high once each, signals an error if value is 
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((assert-in-range pct 0 100) . (LET ((#:G1 PCT) (#:G2 0) (#:G3 100)) (UNLESS (<= #:G2 #:G1 #:G3) (ERROR "~A not in [~A,~A]" #:G1 #:G2 #:G3)) #:G1)))
