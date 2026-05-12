;;; tests.lisp — generated from j14i/cl-ds row 166
;;; instruction: Write a Common Lisp macro `must-be-string` that evaluates an expression once and uses CHECK-TYPE to require the value to
;;; category: validation | technique: gensym | complexity: basic
;;; quality_score: 1.0

(((must-be-string (get-name)) . (LET ((#:G1 (GET-NAME))) (CHECK-TYPE #:G1 STRING) #:G1)))
