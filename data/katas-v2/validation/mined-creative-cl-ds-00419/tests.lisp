;;; tests.lisp — generated from j14i/cl-ds row 419
;;; instruction: Write a Common Lisp macro `must-be` that asserts a value satisfies a unary predicate. The value is evaluated once; an er
;;; category: validation | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((must-be stringp name) . (LET ((#:G1 NAME)) (UNLESS (STRINGP #:G1) (ERROR "~S does not satisfy ~S" #:G1 'STRINGP)) #:G1)))
