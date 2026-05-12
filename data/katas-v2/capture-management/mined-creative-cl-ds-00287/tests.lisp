;;; tests.lisp — generated from j14i/cl-ds row 287
;;; instruction: Write a Common Lisp macro `if-positive` that evaluates a numeric expression; if positive, runs the then-form, else runs 
;;; category: capture-management | technique: once-only | complexity: basic
;;; quality_score: 1.0

(((if-positive (balance) :ok :overdrawn) . (LET ((#:G1 (BALANCE))) (IF (PLUSP #:G1) :OK :OVERDRAWN))))
