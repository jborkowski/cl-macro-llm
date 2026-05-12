;;; tests.lisp — generated from j14i/cl-ds row 240
;;; instruction: Write a Common Lisp macro `inspect-when` that evaluates a value once, calls INSPECT on it only when a test is true (e.g.
;;; category: debugging | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((inspect-when *debug-mode* (find-record id)) . (LET ((#:G1 (FIND-RECORD ID))) (WHEN *DEBUG-MODE* (INSPECT #:G1)) #:G1)))
