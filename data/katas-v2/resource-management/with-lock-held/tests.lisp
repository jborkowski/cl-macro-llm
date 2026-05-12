;;; tests.lisp — hand-crafted, CL idiom
;;; instruction: Write `with-lock-held` — acquire/release pattern via
;;; UNWIND-PROTECT so the lock is released even on non-local exits.
;;; category: resource-management | technique: unwind-protect | complexity: intermediate
;;; quality_score: 0.95

(
 ((with-lock-held (*db-lock*) (write-row row))
  . (UNWIND-PROTECT
         (PROGN (ACQUIRE-LOCK *DB-LOCK*) (WRITE-ROW ROW))
      (RELEASE-LOCK *DB-LOCK*)))

 ((with-lock-held (mutex) (incf counter) (log :tick))
  . (UNWIND-PROTECT
         (PROGN (ACQUIRE-LOCK MUTEX) (INCF COUNTER) (LOG :TICK))
      (RELEASE-LOCK MUTEX)))

 ((with-lock-held ((slot-value obj 'lock)) (mutate obj))
  . (UNWIND-PROTECT
         (PROGN (ACQUIRE-LOCK (SLOT-VALUE OBJ 'LOCK)) (MUTATE OBJ))
      (RELEASE-LOCK (SLOT-VALUE OBJ 'LOCK))))
)
