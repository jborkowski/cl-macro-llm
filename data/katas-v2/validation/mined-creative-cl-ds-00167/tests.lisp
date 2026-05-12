;;; tests.lisp — generated from j14i/cl-ds row 167
;;; instruction: Write a Common Lisp macro `check-keys` that takes a property-list expression and a list of required keyword keys; the pl
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((check-keys options :name :type) . (LET ((#:G1 OPTIONS)) (UNLESS (GETF #:G1 :NAME) (ERROR "Missing required key: ~S" :NAME)) (UNLESS (GETF #:G1 :TYPE) (ERROR "Missing required key: ~S" :TYPE)) #:G1)))
