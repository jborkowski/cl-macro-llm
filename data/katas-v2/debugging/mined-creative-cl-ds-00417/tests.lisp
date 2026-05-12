;;; tests.lisp — generated from j14i/cl-ds row 417
;;; instruction: Write a Common Lisp macro `count-calls-to` that temporarily replaces a function with a counting wrapper during execution
;;; category: debugging | technique: closure-capture | complexity: advanced
;;; quality_score: 1.0

(((count-calls-to sqrt (mapcar #'sqrt '(1 4 9))) . (LET ((#:G1 0)) (LET ((#:G2 (SYMBOL-FUNCTION 'SQRT))) (UNWIND-PROTECT (PROGN (SETF (SYMBOL-FUNCTION 'SQRT) (LAMBDA (&REST #:G3) (INCF #:G1) (APPLY #:G2 #:G3))) (MAPCAR #'SQRT '(1 4 9)) #:G1) (SETF (SYMBOL-FUNCTION 'SQRT) #:G2))))))
