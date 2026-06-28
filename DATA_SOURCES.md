# DATA_SOURCES.md — LOGOS Corpus Provenance

Record every text, transcript, or edition before it enters `data/logos/`.  
**Owner training approval** does not replace a license row — both are required for ingest.

Raw files live in `data/raw/` (gitignored). See `data/raw/README.md` for inventory.

## Status legend

| Status | Meaning |
|--------|---------|
| `approved-train` | Owner confirmed training use |
| `licensed` | Owner-provided file on disk; personal/educational use |
| `licensed — ingested` | Licensed file ingested into `data/logos/` |
| `pd` | Public domain verified for chosen edition |
| `pending` | Not yet verified or not yet provided |
| `hold` | Do not ingest |
| `wrong_edition` | File present but not the intended primary source |

---

## Owner decisions (2026-06-28)

| Decision | Owner |
|----------|-------|
| Train on Osho, Prashant, Chattopadhyaya (Charvaka) | Yes |
| Four scriptures: **split policy** — Bible/Qur'an PD fetch; Gita/GGS owner PDFs | Yes |
| Corpus language: **English-primary** (no Hindi token gate) | Yes |
| **Gita:** TTD 2023 PDF (not Arnold / not Telang primary) | Yes |
| **GGS:** Sant Singh Khalsa PDF (not Macauliffe primary) | Yes |

---

## Owner-provided — licensed (`data/raw/`)

License note: files supplied by Vinayak from personal copies. Record format only in repo; no file contents committed.

### Atheist / secular

| `source_id` | Work | Author | Status | File |
|-------------|------|--------|--------|------|
| `dawkins_god_delusion` | *The God Delusion* | Richard Dawkins | `licensed — ingested` | `data/raw/atheist/dawkins_the_god_delusion.epub` |
| `dawkins_blind_watchmaker` | *The Blind Watchmaker* | Richard Dawkins | `licensed — ingested` | `data/raw/atheist/dawkins_blind_watchmaker.epub` |
| `hitchens_god_not_great` | *God Is Not Great* | Christopher Hitchens | `licensed — ingested` | `data/raw/atheist/hitchens_god_is_not_great.pdf` |
| `hitchens_portable_atheist` | *The Portable Atheist* | Christopher Hitchens (ed.) | `licensed — ingested` | `data/raw/atheist/hitchens_portable_atheist.epub` |
| `harris_end_of_faith` | *The End of Faith* | Sam Harris | `licensed — ingested` | `data/raw/atheist/harris_the_end_of_faith.epub` |
| `dennett_breaking_spell` | *Breaking the Spell* | Daniel Dennett | `licensed — ingested` | `data/raw/atheist/dennett_breaking_the_spell.pdf` |
| `boghossian_manual_atheists` | *A Manual for Creating Atheists* | Peter Boghossian | `licensed — ingested` | `data/raw/atheist/boghossian_manual_creating_atheists.epub` |
| `harris_letter_christian_nation` | *Letter to a Christian Nation* | Sam Harris | `licensed — ingested` | `data/raw/atheist/harris_letter_to_a_christian_nation.pdf` |
| `smith_case_against_god` | *Atheism: The Case Against God* | George H. Smith | `licensed — ingested` | `data/raw/atheist/smith_atheism_case_against_god.pdf` |

### Theist philosophy

| `source_id` | Work | Author | Status | File |
|-------------|------|--------|--------|------|
| `plantinga_gfe` | *God, Freedom, and Evil* | Alvin Plantinga | `licensed — ingested` | `data/raw/theist/plantinga_god_freedom_and_evil.epub` |
| `ghazali_incoherence` | *The Incoherence of the Philosophers* | al-Ghazali | `licensed — ingested` | `data/raw/theist/ghazali_incoherence_of_the_philosophers.pdf` |
| `lewis_mere_christianity` | *Mere Christianity* | C. S. Lewis | `pending` | — |
| `ibn_rushd_incoherence2` | *The Incoherence of the Incoherence* | Ibn Rushd | `licensed — ingested` | `data/raw/theist/ibn_rushd_incoherence_of_the_incoherence.pdf` |

### Osho (`tradition: critique_of_religion`)

| `source_id` | Work | Status | File |
|-------------|------|--------|------|
| `osho_god_is_dead` | *God Is Dead: Now Zen Is the Only Living Truth* | `licensed — ingested` | `data/raw/osho/osho_god_is_dead_zen_only_living_truth.pdf` |
| `osho_the_rebel` | *The Rebel* | `licensed — ingested` | `data/raw/osho/osho_the_rebel.epub` |
| `osho_book_understanding` | *The Book of Understanding* | `licensed — ingested` | `data/raw/osho/osho_book_of_understanding.epub` |

### Buddhism

| `source_id` | Work | Author | Status | File |
|-------------|------|--------|--------|------|
| `suzuki_zen_mind` | *Zen Mind, Beginner's Mind* | Shunryu Suzuki | `licensed — ingested` | `data/raw/buddhism/suzuki_zen_mind_beginners_mind.epub` |
| `dalai_lama_ethics` | *Ethics for the New Millennium* | Dalai Lama XIV | `licensed — ingested` | `data/raw/buddhism/dalai_lama_ethics_for_the_new_millennium.epub` |
| `rahula_what_buddha_taught` | *What the Buddha Taught* | Walpola Rahula | `licensed — ingested` | `data/raw/buddhism/rahula_what_the_buddha_taught.pdf` |

### Indian voices

| `source_id` | Work | Author | Status | File |
|-------------|------|--------|--------|------|
| `bhagat_singh_wiaa` | *Why I Am an Atheist* | Bhagat Singh | `licensed — ingested` | `data/raw/indian/bhagat_singh_why_i_am_an_atheist.pdf` |
| `gandhi_my_religion` | *My Religion* | Gandhi | `licensed — ingested` | `data/raw/indian/gandhi_my_religion.pdf` |
| `chattopadhyaya_charvaka` | *Indian Materialism* / *Philosophy of Charvaka* | Debiprasad Chattopadhyaya | `pending` | — |
| `ambedkar_annihilation_caste` | *Annihilation of Caste* | B. R. Ambedkar | `pending` | — |
| `ambedkar_buddha_dhamma` | *The Buddha and His Dhamma* | B. R. Ambedkar | `pending` | — |

### Transcripts (English-primary)

| `source_id` | Work | Speakers | Status | File |
|-------------|------|----------|--------|------|
| `debate_akhtar_nadwi` | *Does God Exist?* | Javed Akhtar, Mufti Shamail Nadwi | `licensed — ingested` | `data/raw/transcripts/akhtar_nadwi_does_god_exist_en.txt` |
| `prashant_debate_critique` | God-debate critique Q&A | Acharya Prashant | `licensed — ingested` | `data/raw/transcripts/prashant_debate_critique_en.txt` |
| `oconnor_bigthink` | Strongest arguments for/against God | Alex O'Connor | `licensed — ingested` | `data/raw/transcripts/oconnor_bigthink_god_arguments_en.txt` |

### Scriptures (owner-provided — licensed)

| `source_id` | Work | Translator / edition | Status | File |
|-------------|------|----------------------|--------|------|
| `gita_ttd_2023` | *Bhagavad Gita* | TTD 2023; based on Raghavendra Teertha *Gita Vivruti*; compiled Dr. Giridhar Boray | `licensed` — ingested | `data/raw/scriptures/bhagavad_gita_ttd_raghavendra_2023.pdf` |
| `ggs_sant_singh_khalsa` | *Siri Guru Granth Sahib* | Dr. Sant Singh Khalsa (English, matching ang pages); SikhNet | `licensed` — ingested | `data/raw/scriptures/guru_granth_sahib_sant_singh_khalsa_en.pdf` |

License note: Gita PDF © 2023 TTD — *All Rights Reserved*; owner-supplied copy for training.
Verify SikhNet / Khalsa translation terms at ingest; record in this file.

---

## Scriptures — PD fetch (Bible, Qur'an) + PD fallbacks (Gita, GGS)

| `source_id` | Scripture | Edition | Status |
|-------------|-----------|---------|--------|
| `bible_kjv` | Bible | KJV (1769) | `pd` — ingested |
| `quran_rodwell` | Qur'an | Rodwell (1861) | `pd` — ingested |
| `gita_telang_1882` | Bhagavad Gita | Telang (1882, SBE Vol. 8) | `pd` — **fallback only** |
| `ggs_macauliffe_1909` | Guru Granth Sahib | Macauliffe (1909) | `pd` — **fallback only** |

**Primary Gita (owner):** `gita_ttd_2023` — 447 pp; Madhva/Dvaita commentary line; tag `school: dvaita`
where Vivruti commentary; `primary_source` for verse text; `commentary` for Teertha exposition.

**Primary GGS (owner):** `ggs_sant_singh_khalsa` — 1430 pp; split by ang markers (`ÓÓÓÓÓ N ÓÓÓÓÓ`).

**Not used:** Edwin Arnold *Song Celestial* (poetic paraphrase).

---

## Public domain classics (fetch at ingest)

| `source_id` | Work | Status | Ingest notes |
|-------------|------|--------|--------------|
| `hume_dialogues_1779` | *Dialogues Concerning Natural Religion* | `pd` — ingested | Gutenberg 4583 |
| `hume_enquiry_1748` | *An Enquiry Concerning Human Understanding* | `pd` — ingested | Gutenberg 9662 |
| `mackie_evil_1955` | "Evil and Omnipotence" | `pd` — ingested | Inline PD essay text |
| `aquinas_summa_god` | *Summa Theologica* (God selections) | `pd` — ingested | Gutenberg 17620 |
| `paley_natural_theology_1802` | *Natural Theology* | `pd` — ingested | Gutenberg 3239 |
| `plato_euthyphro` | *Euthyphro* | `pd` — ingested | Gutenberg 1657 |
| `aristotle_nicomachean` | *Nicomachean Ethics* | `pd` — ingested | Gutenberg 8438 |
| `mill_utilitarianism_1863` | *Utilitarianism* | `pd` — ingested | Gutenberg 11224 |
| `kant_groundwork` | *Groundwork* | `pd` — ingested | Gutenberg 5682 |
| `upanishads_muller` | *The Upanishads* (Müller) | `pd` — ingested | Gutenberg 3283 |
| `jain_tattvartha` | *Tattvārtha Sūtra* (selections) | `pd` — ingested | Inline PD excerpt |
| `maimonides_guide` | *Guide for the Perplexed* (selections) | `pd` — ingested | Gutenberg 14212 |
| `dhammapada` | *Dhammapada* | `pd` — ingested | Gutenberg 2017 |
| `sutta_anatta` | Anattā-lakkhana Sutta | `pd` — ingested | Inline PD translation |
| `sutta_brahmajala` | Brahmajala Sutta (excerpt) | `pd` — ingested | Inline PD translation |
| `nagarjuna_madhyamaka` | *Mūlamadhyamakakārikā* | `pd` — ingested | Gutenberg 61763 |
| `shantideva_bodhicaryavatara` | *Bodhicaryāvatāra* | `pd` — ingested | Gutenberg 65238 |
| `mumonkan` | *Mumonkan* (selections) | `pd` — ingested | Inline PD koan excerpt |
| `bible_kjv` | Bible (KJV) | `pd` — ingested | Gutenberg 10 |
| `quran_rodwell` | Qur'an (Rodwell) | `pd` — ingested | Gutenberg 2800 |

Hume, Aquinas, Paley, Mackie, Plato, Aristotle, Mill, Kant, Upanishads, Jain Tattvārtha,
Theravada suttas, Nāgārjuna, Shantideva — see Phase 9 plan and `configs/source_registry.yaml`.
