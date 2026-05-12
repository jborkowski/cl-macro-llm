;;; tests.lisp — generated from j14i/cl-ds row 511
;;; instruction: Write a Common Lisp macro `with-timer-thread` that spawns a periodic timer thread via `start-timer-thread` running a use
;;; category: resource-management | technique: gensym | complexity: intermediate
;;; quality_score: 1.0

(((with-timer-thread (tt 1.0 (heartbeat)) (long-task)) . (LET ((#:G1 (START-TIMER-THREAD 1.0 (LAMBDA () (HEARTBEAT))))) (LET ((TT #:G1)) (UNWIND-PROTECT (PROGN (LONG-TASK)) (KILL-THREAD #:G1))))))
