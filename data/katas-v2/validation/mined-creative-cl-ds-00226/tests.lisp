;;; tests.lisp — generated from j14i/cl-ds row 226
;;; instruction: Write a Common Lisp macro `assert-bounded` that signals an error if a numeric value is not within an inclusive range [lo
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((assert-bounded score 0 100) . (LET ((#:G1 SCORE)) (UNLESS (AND (>= #:G1 0) (<= #:G1 100)) (ERROR "Value ~S not in [~S, ~S]" #:G1 0 100)) #:G1)))
