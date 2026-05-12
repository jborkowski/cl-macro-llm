;;; tests.lisp — generated from j14i/cl-ds row 245
;;; instruction: Write a Common Lisp macro `with-printed-vars` that evaluates a body, then prints the current value of each named variabl
;;; category: debugging | technique: dispatch | complexity: intermediate
;;; quality_score: 1.0

(((with-printed-vars (x y) (setf x 1 y 2)) . (LET ((#:G1 (PROGN (SETF X 1 Y 2)))) (FORMAT *TRACE-OUTPUT* "~&~A = ~S~%" (QUOTE X) X) (FORMAT *TRACE-OUTPUT* "~&~A = ~S~%" (QUOTE Y) Y) #:G1)))
