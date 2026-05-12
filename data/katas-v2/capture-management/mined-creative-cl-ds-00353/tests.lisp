;;; tests.lisp — generated from j14i/cl-ds row 353
;;; instruction: Write a Common Lisp macro `with-saved` that captures the current value of a place into a hidden gensym, runs body, and r
;;; category: capture-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-saved counter (incf counter) (process)) . (LET ((#:G1 COUNTER)) (UNWIND-PROTECT (PROGN (INCF COUNTER) (PROCESS)) (SETF COUNTER #:G1)))))
