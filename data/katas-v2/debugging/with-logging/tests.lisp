;;; tests.lisp — hand-crafted, CL idiom
;;; instruction: Write `with-logging` — emit entry/exit log lines
;;; around a body.
;;; category: debugging | technique: trace-wrap | complexity: basic
;;; quality_score: 0.95

(
 ((with-logging :boot (load-config) (start-server))
  . (PROGN
      (FORMAT T "[enter ~A]~%" ':BOOT)
      (LOAD-CONFIG)
      (START-SERVER)
      (FORMAT T "[exit ~A]~%" ':BOOT)))

 ((with-logging "compile-step" (analyze) (emit))
  . (PROGN
      (FORMAT T "[enter ~A]~%" '"compile-step")
      (ANALYZE)
      (EMIT)
      (FORMAT T "[exit ~A]~%" '"compile-step")))

 ((with-logging shutdown (flush-buffers) (close-streams))
  . (PROGN
      (FORMAT T "[enter ~A]~%" 'SHUTDOWN)
      (FLUSH-BUFFERS)
      (CLOSE-STREAMS)
      (FORMAT T "[exit ~A]~%" 'SHUTDOWN)))
)
