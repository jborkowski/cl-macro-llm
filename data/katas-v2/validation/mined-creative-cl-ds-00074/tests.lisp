;;; tests.lisp — generated from j14i/cl-ds row 74
;;; instruction: Write a Common Lisp macro `require-keys` that evaluates a plist expression once and signals an error for any listed key 
;;; category: validation | technique: dispatch | complexity: intermediate
;;; quality_score: 1.0

(((require-keys opts :name :id) . (LET ((#:G1 OPTS)) (UNLESS (GETF #:G1 :NAME) (ERROR "Missing required key ~S" :NAME)) (UNLESS (GETF #:G1 :ID) (ERROR "Missing required key ~S" :ID)) #:G1)))
