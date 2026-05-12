;;; tests.lisp — generated from j14i/cl-ds row 497
;;; instruction: Write a Common Lisp macro `require-all` that evaluates a value once and signals an error if it fails any of the given pr
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((require-all x numberp plusp integerp) . (LET ((#:G1 X)) (UNLESS (NUMBERP #:G1) (ERROR "~A failed ~A" #:G1 'NUMBERP)) (UNLESS (PLUSP #:G1) (ERROR "~A failed ~A" #:G1 'PLUSP)) (UNLESS (INTEGERP #:G1) (ERROR "~A failed ~A" #:G1 'INTEGERP)) #:G1)))
