;;; tests.lisp — generated from j14i/cl-ds row 307
;;; instruction: Write a Common Lisp macro `with-timing` that executes body, prints the labeled elapsed wall-clock time to *trace-output*
;;; category: debugging | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-timing "task" (foo) (bar)) . (LET ((#:G1 (GET-INTERNAL-REAL-TIME))) (LET ((#:G2 (PROGN (FOO) (BAR)))) (FORMAT *TRACE-OUTPUT* "~&[~A] ~,3F seconds~%" "task" (FLOAT (/ (- (GET-INTERNAL-REAL-TIME) #:G1) INTERNAL-TIME-UNITS-PER-SECOND))) #:G2))))
