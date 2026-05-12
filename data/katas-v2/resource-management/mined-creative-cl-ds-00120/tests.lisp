;;; tests.lisp — generated from j14i/cl-ds row 120
;;; instruction: Write a Common Lisp macro `with-saved-random-state` that snapshots *random-state* (via make-random-state to copy it) bef
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-saved-random-state (random 10) (random 10)) . (LET ((#:G1 (MAKE-RANDOM-STATE *RANDOM-STATE*))) (UNWIND-PROTECT (PROGN (RANDOM 10) (RANDOM 10)) (SETF *RANDOM-STATE* #:G1)))))
