// Package stemmer подключает стеммеры и стоп-слова для BM25-токенизатора.
//
// Каждый язык регистрируется с именем и одной реализацией Stemmer, которая сама
// заявляет свою письменность (Claims), распознаёт стоп-слова (IsStopword) и
// сводит токен к основе (Stem). Реестр позволяет добавлять и убирать языки
// «малой кровью»: новый язык — это файл с Register в init(), удаление — удалить
// этот файл, после чего токены его письменности просто проходят насквозь без
// изменений. Ядро поиска не знает имён языков и конкретных алгоритмов.
//
// Select ограничивает стемминг подмножеством языков: пустой список — все
// зарегистрированные (автовыбор по письменности), список имён — только эти
// языки, остальные токены проходят насквозь. Это даёт параметр «язык стеммера»
// в редакторе/конфиге сервера при сохранении обратной совместимости по умолчанию.
package stemmer

import (
	"fmt"
	"strings"
)

// Stemmer описывает один язык: детект письменности, стоп-слова и сведение
// токена к основе. Все методы вызываются с одним и тем же токеном.
type Stemmer interface {
	// Claims сообщает, принадлежит ли токен письменности языка. Токены, не
	// заявленные ни одним зарегистрированным стеммером, проходят насквозь
	// без изменений.
	Claims(token string) bool

	// Stem сводит токен к морфологической основе. Пустая строка означает,
	// что токен должен быть отброшен.
	Stem(token string) string

	// IsStopword сообщает, является ли токен стоп-словом языка: такой токен
	// отбрасывается целиком.
	IsStopword(token string) bool
}

// namedStemmer связывает зарегистрированный язык с его реализацией.
type namedStemmer struct {
	name string
	s    Stemmer
}

// registered хранит стеммеры в порядке регистрации; при пересечении Claims
// побеждает первый. Список заполняется только в init(), поэтому чтение во
// время работы не требует синхронизации.
var registered []namedStemmer

// Register добавляет стеммер в реестр под именем name. Обычно вызывается из
// init пакета конкретного языка.
func Register(name string, s Stemmer) {
	registered = append(registered, namedStemmer{name: name, s: s})
}

// Names возвращает имена зарегистрированных стеммеров в порядке регистрации.
func Names() []string {
	out := make([]string, len(registered))
	for i, e := range registered {
		out[i] = e.name
	}
	return out
}

// resolve возвращает стеммер, заявивший токен, или false, если такого нет.
func resolve(token string) (Stemmer, bool) {
	for _, e := range registered {
		if e.s.Claims(token) {
			return e.s, true
		}
	}
	return nil, false
}

// StemToken прогоняет один токен через активный стеммер: стоп-слово или
// пустая основа дают "", токены без заявившего их языка возвращаются как есть.
func StemToken(token string) string {
	s, ok := resolve(token)
	if !ok {
		return token
	}
	if s.IsStopword(token) {
		return ""
	}
	return s.Stem(token)
}

// Select возвращает функцию стемминга токена для перечисленных языков. Пустой
// список (или список из одних пустых имён) означает «все языки» — то есть
// текущее автовыборное поведение StemToken. Неизвестное имя — ошибка, чтобы
// опечатку в конфиге можно было отловить легально, а не молчаливо отключить
// стемминг.
func Select(langs []string) (func(string) string, error) {
	if len(langs) == 0 {
		return StemToken, nil
	}
	byName := make(map[string]Stemmer, len(registered))
	for _, e := range registered {
		byName[e.name] = e.s
	}
	chosen := make([]Stemmer, 0, len(langs))
	var unknown []string
	for _, l := range langs {
		l = strings.ToLower(strings.TrimSpace(l))
		if l == "" {
			continue
		}
		if s, ok := byName[l]; ok {
			chosen = append(chosen, s)
		} else {
			unknown = append(unknown, l)
		}
	}
	if len(unknown) > 0 {
		return nil, fmt.Errorf("stemming language(s) %v not registered (available: %v)", unknown, Names())
	}
	if len(chosen) == 0 {
		return StemToken, nil
	}
	return func(token string) string {
		for _, s := range chosen {
			if s.Claims(token) {
				if s.IsStopword(token) {
					return ""
				}
				return s.Stem(token)
			}
		}
		return token
	}, nil
}

// HasCyrillic сообщает, содержит ли строка кириллические буквы.
func HasCyrillic(s string) bool {
	for _, r := range s {
		if r >= 0x0400 && r <= 0x04FF {
			return true
		}
	}
	return false
}
