// Package stemmer подключает стеммеры и стоп-слова для BM25-токенизатора.
//
// Каждый язык регистрирует одну реализацию Stemmer, которая сама заявляет
// свою письменность (Claims), распознаёт стоп-слова (IsStopword) и сводит
// токен к основе (Stem). Реестр позволяет добавлять и убирать языки «малой
// кровью»: новый язык — это файл с Register в init(), удаление — удалить этот
// файл, после чего токены его письменности просто проходят насквозь без
// изменений. Ядро поиска не знает имён языков и конкретных алгоритмов.
package stemmer

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

// registered хранит стеммеры в порядке регистрации; при пересечении Claims
// побеждает первый. Список заполняется только в init(), поэтому чтение во
// время работы не требует синхронизации.
var registered []Stemmer

// Register добавляет стеммер в реестр. Обычно вызывается из init пакета
// конкретного языка.
func Register(s Stemmer) {
	registered = append(registered, s)
}

// resolve возвращает стеммер, заявивший токен, или false, если такого нет.
func resolve(token string) (Stemmer, bool) {
	for _, s := range registered {
		if s.Claims(token) {
			return s, true
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

// HasCyrillic сообщает, содержит ли строка кириллические буквы.
func HasCyrillic(s string) bool {
	for _, r := range s {
		if r >= 0x0400 && r <= 0x04FF {
			return true
		}
	}
	return false
}
