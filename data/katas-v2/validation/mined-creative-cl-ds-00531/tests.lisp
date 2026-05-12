;;; tests.lisp — generated from j14i/cl-ds row 531
;;; instruction: Write a Common Lisp macro `ensure-list-of` that evaluates a list-producing expression once and asserts that every elemen
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((ensure-list-of integerp (get-nums)) . (LET ((#:G1 (GET-NUMS))) (DOLIST (#:G2 #:G1) (UNLESS (INTEGERP #:G2) (ERROR "Element ~S fails ~S" #:G2 'INTEGERP))) #:G1)))
