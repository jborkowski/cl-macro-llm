;;; tests.lisp — generated from j14i/cl-ds row 230
;;; instruction: Write a Common Lisp macro `assert-divides` that signals an error if a divisor does not evenly divide a value; otherwise 
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((assert-divides 3 amount) . (LET ((#:G1 3) (#:G2 AMOUNT)) (UNLESS (ZEROP (MOD #:G2 #:G1)) (ERROR "~S does not divide ~S" #:G1 #:G2)) #:G2)))
