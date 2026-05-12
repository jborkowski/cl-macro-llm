;;; tests.lisp — generated from j14i/cl-ds row 529
;;; instruction: Write a Common Lisp macro `must-satisfy` that evaluates an expression once, signals a descriptive error if the value doe
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((must-satisfy plusp (compute-value)) . (LET ((#:G1 (COMPUTE-VALUE))) (UNLESS (PLUSP #:G1) (ERROR "~S failed predicate ~S" #:G1 'PLUSP)) #:G1)))
