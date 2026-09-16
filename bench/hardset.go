package bench

import (
	"github.com/Gaurav-Gosain/turbograph/eval"
	"github.com/Gaurav-Gosain/turbograph/rag"
)

// HardDataset returns a synthetic, deterministic corpus with labeled queries that
// are genuinely hard for a lexical-only or dense-only retriever, built to run
// offline in CI with HashEmbedder. Unlike goldDataset, where every query is the
// clean lexical match of one document, these cases fail a naive top-k baseline and
// only the full pipeline (hybrid fusion, similarity-graph boosting, chunk-level
// dedup and document collapsing) retrieves the ground truth reliably. The score on
// this set is the regression canary: a ranking change that helps one family at the
// expense of another shows up here before it reaches a real corpus.
//
// Every document and query is authored with distinct vocabulary, so the HashEmbedder
// (bag-of-words, no semantics) reproduces exactly the overlap patterns below and the
// whole suite repeats bit-for-bit in CI. The families:
//
//   - "multi_hop": the ground truth is a pair of documents that no single query
//     sentence connects. The answer text lives in one document; the second document
//     is only reachable through a rare shared token and the chunk-similarity graph.
//     Without GraphMix the pair is only half retrieved (recall 0.5, MRR halved).
//   - "near_miss": within-domain decoys. The decoy shares the query's common terms
//     verbatim and differs from the relevant document by one discriminating fact
//     term. Dense ranking puts both at the top; fusing BM25 is what keeps the
//     relevant one first.
//   - "distraught": the answer is spread across two chunks of one document while
//     decoys replicate one of the query halves almost exactly. Winning means the
//     document survives chunk collapsing with all of its evidence intact.
func HardDataset() (*Dataset, []string) {
	docs := []rag.Document{
		// --- multi_hop cluster 1: Titan / Acme Robotics ---
		{ID: "titan-bridge", Text: "Titan Industries merged with Acme Robotics in March and the combined firm kept the Acme name."},
		{ID: "acme-answer", Text: "Precision surgical robots for operating rooms are built by Acme Robotics."},
		{ID: "titan-filler", Text: "The harbour crane lifts containers from the container ship."},
		// --- multi_hop cluster 2: Helios / Nova battery ---
		{ID: "helios-bridge", Text: "Helios acquired the Nova startup and moved it to branch six."},
		{ID: "nova-answer", Text: "The Nova battery charges fully in minutes using a graphene cathode."},
		{ID: "helios-filler", Text: "Wind turbines yaw to face the prevailing breeze."},
		// --- multi_hop cluster 3: Quarry / Lark network ---
		{ID: "quarry-bridge", Text: "The Quarry studio partnered with Lark in its second year."},
		{ID: "lark-answer", Text: "Lark streaming encodes high bitrate video with a custom codec."},
		{ID: "quarry-filler", Text: "Documentaries cut between interviews and archival footage."},
		// --- multi_hop cluster 4: Basin / Cohort census ---
		{ID: "basin-bridge", Text: "The Basin hospital group shares patient data with the Cohort census project."},
		{ID: "cohort-answer", Text: "The Cohort census profiles chronic conditions across five million residents."},
		{ID: "basin-filler", Text: "Cardiac units monitor rhythm with wearable patches."},
		// --- multi_hop cluster 5: Hamilton / Apollo reactor ---
		{ID: "hamilton-bridge", Text: "Apollo fuel rods emerged from the Hamilton naval division."},
		{ID: "apollo-answer", Text: "Apollo fuel cells power a compact reactor tested on the bench."},
		{ID: "hamilton-filler", Text: "Frigates dry dock annually for hull inspection."},

		// --- near_miss domain 1: tomato ---
		{ID: "tomato-correct", Text: "The ordinary tomato is classified a berry because the ovary wall turns fleshy."},
		{ID: "tomato-decoy", Text: "An 1893 ruling treated the tomato as a vegetable for tariff purposes."},
		// --- near_miss domain 2: eclipses ---
		{ID: "eclipse-correct", Text: "A solar eclipse requires the moon to align between the sun and the earth."},
		{ID: "eclipse-decoy", Text: "A lunar eclipse happens when the earth shadows the moon."},
		// --- near_miss domain 3: bitcoin ---
		{ID: "bitcoin-correct", Text: "Proof of work secures the bitcoin ledger by making fraud expensive."},
		{ID: "bitcoin-decoy", Text: "Bitcoin price swings mirror trading volumes on exchanges."},
		// --- near_miss domain 4: glaciers ---
		{ID: "glacier-correct", Text: "Glaciers retreat when ablation exceeds annual snow accumulation."},
		{ID: "glacier-decoy", Text: "Glaciers carve U shaped valleys as they grind downhill."},
		// --- near_miss domain 5: bees ---
		{ID: "bee-correct", Text: "Honey bees communicate distance and direction through the waggle dance."},
		{ID: "bee-decoy", Text: "Honey bees collect nectar and pollen from flowering plants."},
		// --- near_miss domain 6: proxies ---
		{ID: "proxy-correct", Text: "A forward proxy hides the client address from the origin server."},
		{ID: "proxy-decoy", Text: "A reverse proxy fronts backend servers to public clients."},

		// --- distraught: answer split across two chunks, decoys mimic one half ---
		{ID: "deep-correct", Text: "Submarine cables cross the ocean floor carrying most intercontinental traffic. Signals are regenerated by repeaters every few thousand kilometres."},
		{ID: "deep-decoy1", Text: "Submarine cables cross the ocean floor laid by specialised ships."},
		{ID: "deep-decoy2", Text: "Repeaters boost every signal passing along long fibre links."},
		{ID: "linen-correct", Text: "Linen is spun from flax fibres and prized for stiff cool fabric. The fibres come from the flax stem after retting in water."},
		{ID: "linen-decoy1", Text: "Linen is spun from flax fibres and woven into tablecloths."},
		{ID: "linen-decoy2", Text: "Flax stems after retting in water produce strong cordage."},
		{ID: "ice-correct", Text: "Ice shelves float at the margin of the ice sheet and buttress inland glaciers. Their loss accelerates the discharge of grounded ice into the sea."},
		{ID: "ice-decoy1", Text: "Ice shelves float at the margin of the ice sheet covering the Antarctic."},
		{ID: "ice-decoy2", Text: "Grounded ice discharges into the sea as calving icebergs."},
		{ID: "loom-correct", Text: "A handloom weaves cloth by beating the weft into the warp. Jacquard attachment uses punched cards to control each warp thread."},
		{ID: "loom-decoy1", Text: "A handloom weaves cloth by beating the weft into the warp."},
		{ID: "loom-decoy2", Text: "Punched cards on a jacquard machine control each warp thread."},

		// --- fillers: vocabulary-noise so a query with few hits is not padded by
		// arbitrary zero-cosine chunks. Without them a dense-only top-k pulls in
		// filler chunks and "recalls" multi-hop bridge documents by accident.
		{ID: "coral", Text: "Coral polyps host zooxanthellae algae that provide energy through photosynthesis."},
		{ID: "saffron", Text: "Saffron is harvested from the stigma of the crocus flower by hand each autumn."},
		{ID: "tides", Text: "Ocean tides are driven by the gravitational pull of the moon and sun."},
		{ID: "velvet", Text: "Velvet is woven on a loom with two layers and cut apart down the middle."},
		{ID: "geese", Text: "Geese migrate along fixed flyways, navigating by the sun, the stars, and landmarks."},
		{ID: "magnet", Text: "A permanent magnet has aligned domains that reinforce one field direction."},
		{ID: "quartz", Text: "Quartz crystals vibrate at a precise frequency when an electric field is applied."},
		{ID: "onion", Text: "Onions store sugars in concentric layers of modified leaf bases."},
		{ID: "prism", Text: "A prism bends each wavelength of white light by a different angle."},
		{ID: "gourd", Text: "Dried gourds make hard vessels because their shell hardens as it matures."},
		{ID: "yarn", Text: "Yarn twists staple fibres together so the strand holds under tension."},
		{ID: "comet", Text: "Comets release gas and dust as they approach the inner solar system."},
		{ID: "ivory", Text: "Ivory carves cleanly with fine detail but cracks when it dries too fast."},
		{ID: "honey", Text: "Honey resists spoilage because of low moisture and high sugar content."},
		{ID: "jade", Text: "Jade is valued for toughness and carved by grinding with abrasives."},
		{ID: "sorbet", Text: "Sorbet is churned ice flavoured with fruit and syrup, with no dairy."},
		{ID: "kayak", Text: "A kayak is paddled with a double bladed paddle from a sealed cockpit."},
		{ID: "truffle", Text: "Truffles grow underground near tree roots and are found by dogs."},
		{ID: "beacon", Text: "Lighthouses project a beacon that is visible far out to sea."},
		{ID: "zinc", Text: "Zinc galvanises steel by corroding sacrificially before the iron does."},
	}

	cases := []eval.Case{
		// multi_hop: the query only shares vocabulary with the answer; the bridge
		// document (same entity, corroborating fact) shares no query token and is
		// reachable only through the chunk-similarity graph edge to the answer.
		{Query: "surgical robots for operating rooms", Relevant: []string{"acme-answer", "titan-bridge"}},
		{Query: "robots built for sterile operating rooms", Relevant: []string{"acme-answer", "titan-bridge"}},
		{Query: "battery that charges in minutes", Relevant: []string{"nova-answer", "helios-bridge"}},
		{Query: "battery with a graphene cathode", Relevant: []string{"nova-answer", "helios-bridge"}},
		{Query: "video encoded with a custom codec", Relevant: []string{"lark-answer", "quarry-bridge"}},
		{Query: "streaming network high bitrate encoding", Relevant: []string{"lark-answer", "quarry-bridge"}},
		{Query: "chronic conditions across five million residents", Relevant: []string{"cohort-answer", "basin-bridge"}},
		{Query: "profiling chronic conditions among residents", Relevant: []string{"cohort-answer", "basin-bridge"}},
		{Query: "compact reactor tested on the power bench", Relevant: []string{"apollo-answer", "hamilton-bridge"}},
		{Query: "compact power cells inside the reactor", Relevant: []string{"apollo-answer", "hamilton-bridge"}},

		// near_miss: the decoy quotes the query's common terms; only the fact term
		// separates relevant from decoy.
		{Query: "is the tomato classified as a berry", Relevant: []string{"tomato-correct"}},
		{Query: "tomato berry versus vegetable ruling", Relevant: []string{"tomato-correct"}},
		{Query: "solar eclipse moon sun alignment", Relevant: []string{"eclipse-correct"}},
		{Query: "eclipse shadow alignment moon earth", Relevant: []string{"eclipse-correct"}},
		{Query: "bitcoin proof of work securing the ledger", Relevant: []string{"bitcoin-correct"}},
		{Query: "proof of work versus price swings", Relevant: []string{"bitcoin-correct"}},
		{Query: "glacier retreat ablation and accumulation", Relevant: []string{"glacier-correct"}},
		{Query: "glacier ablation carving the valley", Relevant: []string{"glacier-correct"}},
		{Query: "honey bees waggle dance communication", Relevant: []string{"bee-correct"}},
		{Query: "bees distance and direction dance", Relevant: []string{"bee-correct"}},
		{Query: "forward proxy hiding the client address", Relevant: []string{"proxy-correct"}},
		{Query: "proxy origin server client address", Relevant: []string{"proxy-correct"}},

		// distraught: answer spans two chunks; decoys quote a single chunk verbatim.
		{Query: "submarine cables ocean floor and signal repeaters", Relevant: []string{"deep-correct"}},
		{Query: "repeaters regenerating submarine cable signals", Relevant: []string{"deep-correct"}},
		{Query: "linen fibres from flax stem after retting", Relevant: []string{"linen-correct"}},
		{Query: "flax retting producing stiff cool fabric", Relevant: []string{"linen-correct"}},
		{Query: "ice shelves buttressing grounded ice discharge", Relevant: []string{"ice-correct"}},
		{Query: "loss of ice shelves accelerating glacier discharge", Relevant: []string{"ice-correct"}},
		{Query: "handloom weft warp and jacquard punched cards", Relevant: []string{"loom-correct"}},
		{Query: "jacquard punched cards weaving the warp", Relevant: []string{"loom-correct"}},
	}

	families := make([]string, len(cases))
	for i := range families {
		switch {
		case i < 10:
			families[i] = "multi_hop"
		case i < 22:
			families[i] = "near_miss"
		default:
			families[i] = "distraught"
		}
	}
	return &Dataset{Name: "hard", Docs: docs, Cases: cases}, families
}
