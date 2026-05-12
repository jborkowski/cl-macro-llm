;;; tests.lisp — generated from j14i/cl-ds row 350
;;; instruction: Write a Common Lisp macro `min-max-let` that evaluates two expressions exactly once each, binds the smaller to a user-na
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((min-max-let (lo hi) (first xs) (car (last xs)) (list lo hi)) . (LET ((#:G1 (FIRST XS)) (#:G2 (CAR (LAST XS)))) (LET ((LO (MIN #:G1 #:G2)) (HI (MAX #:G1 #:G2))) (LIST LO HI)))))
