;;; tests.lisp — generated from j14i/cl-ds row 289
;;; instruction: Write a Common Lisp macro `do-times-by` that runs body with an internal counter going from 0 up to n exclusive, advancin
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((do-times-by 10 2 (print :tick)) . (LET ((#:G1 10) (#:G2 2)) (DO ((#:G3 0 (+ #:G3 #:G2))) ((>= #:G3 #:G1)) (PRINT :TICK)))))
