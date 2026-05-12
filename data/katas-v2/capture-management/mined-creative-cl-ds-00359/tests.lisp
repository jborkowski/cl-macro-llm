;;; tests.lisp — generated from j14i/cl-ds row 359
;;; instruction: Write a Common Lisp macro `safe-sqrt` that returns 0 when the argument is negative and the square root otherwise, evalua
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((safe-sqrt (compute)) . (LET ((#:G1 (COMPUTE))) (IF (MINUSP #:G1) 0 (SQRT #:G1)))))
