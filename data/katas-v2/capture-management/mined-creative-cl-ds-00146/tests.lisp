;;; tests.lisp — generated from j14i/cl-ds row 146
;;; instruction: Write a Common Lisp macro `with-accumulator` that hides a numeric accumulator cell and binds a user-named local function
;;; category: capture-management | technique: closure-capture | complexity: intermediate
;;; quality_score: 1.0

(((with-accumulator (add 0) (add 5) (add 7)) . (LET ((#:G1 0)) (FLET ((ADD (#:G2) (INCF #:G1 #:G2))) (ADD 5) (ADD 7) #:G1))))
