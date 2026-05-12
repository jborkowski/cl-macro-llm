;;; tests.lisp — generated from j14i/cl-ds row 161
;;; instruction: Write a Common Lisp macro `ensure-non-nil` that evaluates an expression once; if the value is nil, signal an error; othe
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((ensure-non-nil (lookup :key)) . (LET ((#:G1 (LOOKUP :KEY))) (IF (NULL #:G1) (ERROR "Value is nil") #:G1))))
