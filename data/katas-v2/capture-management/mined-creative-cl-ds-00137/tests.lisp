;;; tests.lisp — generated from j14i/cl-ds row 137
;;; instruction: Write a Common Lisp macro `mulf` that multiplies a place by a factor in place. The factor expression must be evaluated e
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((mulf x (compute)) . (LET ((#:G1 (COMPUTE))) (SETF X (* X #:G1)))))
