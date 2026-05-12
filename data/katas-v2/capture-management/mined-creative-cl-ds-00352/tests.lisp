;;; tests.lisp — generated from j14i/cl-ds row 352
;;; instruction: Write a Common Lisp macro `nth-bind` that evaluates a list expression once and binds its first three elements to user-na
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((nth-bind (x y z) (get-vec) (+ x y z)) . (LET ((#:G1 (GET-VEC))) (LET ((X (FIRST #:G1)) (Y (SECOND #:G1)) (Z (THIRD #:G1))) (+ X Y Z)))))
