;;; tests.lisp — generated from j14i/cl-ds row 203
;;; instruction: Write a Common Lisp macro `nif` (numeric if) that evaluates an expression once and dispatches to one of three branches b
;;; category: capture-management | technique: once-only | complexity: intermediate
;;; quality_score: 1.0

(((nif (- a b) :positive :equal :negative) . (LET ((#:G1 (- A B))) (COND ((PLUSP #:G1) :POSITIVE) ((ZEROP #:G1) :EQUAL) (T :NEGATIVE)))))
