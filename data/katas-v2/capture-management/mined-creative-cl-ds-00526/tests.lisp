;;; tests.lisp — generated from j14i/cl-ds row 526
;;; instruction: Write a Common Lisp macro `make-counter` that expands into a closure starting at an optional initial value (default 0). 
;;; category: capture-management | technique: closure-capture | complexity: intermediate
;;; quality_score: 1.0

(((make-counter 10) . (LET ((#:G1 10)) (LAMBDA () (INCF #:G1)))))
