;;; tests.lisp — generated from j14i/cl-ds row 491
;;; instruction: Write a Common Lisp macro `define-memoized` that defines a function with a private cache hash-table. Repeated calls with
;;; category: capture-management | technique: closure-capture | complexity: advanced
;;; quality_score: 1.0

(((define-memoized slow-fn (n) (* n n)) . (LET ((#:G1 (MAKE-HASH-TABLE :TEST 'EQUAL))) (DEFUN SLOW-FN (N) (LET ((#:G2 (LIST N))) (MULTIPLE-VALUE-BIND (#:G3 #:G4) (GETHASH #:G2 #:G1) (IF #:G4 #:G3 (SETF (GETHASH #:G2 #:G1) (PROGN (* N N))))))))))
