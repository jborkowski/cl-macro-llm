;;; tests.lisp — generated from j14i/cl-ds row 465
;;; instruction: Write a Common Lisp macro `with-call-counter` that introduces a local function which, each time it is called, increments
;;; category: debugging | technique: gensym | complexity: advanced
;;; quality_score: 1.0

(((with-call-counter (tick) (dotimes (i 5) (tick))) . (LET ((#:G1 0)) (FLET ((TICK NIL (INCF #:G1))) (DOTIMES (I 5) (TICK))) (FORMAT *TRACE-OUTPUT* "~&~A called ~D times~%" 'TICK #:G1))))
