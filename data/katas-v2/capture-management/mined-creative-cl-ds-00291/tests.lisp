;;; tests.lisp — generated from j14i/cl-ds row 291
;;; instruction: Write a Common Lisp macro `with-saved-place` that snapshots a place's current value, executes body, and restores the pla
;;; category: capture-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-saved-place *db* (setf *db* :test) (run-tests)) . (LET ((#:G1 *DB*)) (UNWIND-PROTECT (PROGN (SETF *DB* :TEST) (RUN-TESTS)) (SETF *DB* #:G1)))))
