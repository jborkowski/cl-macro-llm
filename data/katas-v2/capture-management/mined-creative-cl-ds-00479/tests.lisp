;;; tests.lisp — generated from j14i/cl-ds row 479
;;; instruction: Write a Common Lisp macro `tee` that evaluates an expression once and feeds the value to two functions, returning both r
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((tee (compute) #'print #'log) . (LET ((#:G1 (COMPUTE))) (VALUES (FUNCALL #'PRINT #:G1) (FUNCALL #'LOG #:G1)))))
