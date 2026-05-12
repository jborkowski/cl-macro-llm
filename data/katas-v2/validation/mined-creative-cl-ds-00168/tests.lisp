;;; tests.lisp — generated from j14i/cl-ds row 168
;;; instruction: Write a Common Lisp macro `ensure-list` that evaluates an expression once; if the value is a list, return it, otherwise 
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((ensure-list (get-items)) . (LET ((#:G1 (GET-ITEMS))) (UNLESS (LISTP #:G1) (ERROR "Expected list, got ~S" #:G1)) #:G1)))
