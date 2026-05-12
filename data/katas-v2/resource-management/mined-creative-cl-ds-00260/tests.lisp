;;; tests.lisp — generated from j14i/cl-ds row 260
;;; instruction: Write a Common Lisp macro `with-saved-variable` that saves the current value of a special variable, sets it to a new val
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-saved-variable (*foo* 42) (use-foo)) . (LET ((#:G1 *FOO*)) (UNWIND-PROTECT (PROGN (SETF *FOO* 42) (USE-FOO)) (SETF *FOO* #:G1)))))
