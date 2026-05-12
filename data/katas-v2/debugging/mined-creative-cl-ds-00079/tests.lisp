;;; tests.lisp — generated from j14i/cl-ds row 79
;;; instruction: Write a Common Lisp macro `time-tagged` that measures the wall-clock duration of running a body, prints the duration wit
;;; category: debugging | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((time-tagged "fetch" (fetch-data)) . (LET ((#:G1 (GET-INTERNAL-REAL-TIME))) (LET ((#:G2 (PROGN (FETCH-DATA)))) (FORMAT *TRACE-OUTPUT* "~&[~A] ~Fs~%" "fetch" (/ (- (GET-INTERNAL-REAL-TIME) #:G1) (FLOAT INTERNAL-TIME-UNITS-PER-SECOND))) #:G2))))
