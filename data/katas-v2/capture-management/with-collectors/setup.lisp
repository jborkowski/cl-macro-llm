;;; setup.lisp — with-collectors, function-capturing accumulator.
;;; KEY HYGIENE NOTE: each name supplied by the caller is captured TWICE
;;; — once as a LET variable (holding the accumulating reversed list)
;;; and once as an FLET function (pushing to that variable). The body
;;; calls (name x) to collect into the list named `name`. Do NOT gensym
;;; the names — they must be the caller's symbols.

(in-package :cl-user)
