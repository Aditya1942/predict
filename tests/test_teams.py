import pytest

from predict.domain import Team
from predict.errors import AmbiguousTeam, SourceError, TeamNotFound
from predict.teams import Aliases, TeamResolver, normalize, parse_query

# (id, football-data.org name, short name, tla)
_TEAMS = [
    (57, "Arsenal FC", "Arsenal", "ARS"),
    (61, "Chelsea FC", "Chelsea", "CHE"),
    (65, "Manchester City FC", "Man City", "MCI"),
    (66, "Manchester United FC", "Man United", "MUN"),
    (397, "Brighton & Hove Albion FC", "Brighton Hove", "BHA"),
    (64, "Liverpool FC", "Liverpool", "LIV"),
    (81, "FC Barcelona", "Barça", "FCB"),
    (86, "Real Madrid CF", "Real Madrid", "RMA"),
    (90, "Real Betis Balompié", "Real Betis", "BET"),
    (92, "Real Sociedad de Fútbol", "Real Sociedad", "RSO"),
    (78, "Club Atlético de Madrid", "Atleti", "ATL"),
    (5, "FC Bayern München", "Bayern M", "FCB"),
    (3, "Bayer 04 Leverkusen", "Leverkusen", "B04"),
    (524, "Paris Saint-Germain FC", "PSG", "PSG"),
    (548, "AS Monaco FC", "Monaco", "ASM"),
    (100, "AS Roma", "AS Roma", "ROM"),
    (108, "FC Internazionale Milano", "Inter", "INT"),
    (98, "AC Milan", "Milan", "MIL"),
    (7397, "Como 1907", "Como", "COM"),
]
TEAMS = [Team(id=i, name=n, short_name=s, tla=t) for i, n, s, t in _TEAMS]
ID = {t.name: t.id for t in TEAMS}
ARSENAL_ID = ID["Arsenal FC"]


@pytest.fixture(scope="module")
def aliases() -> Aliases:
    return Aliases.load()


@pytest.fixture(scope="module")
def resolver(aliases) -> TeamResolver:
    return TeamResolver(TEAMS, aliases)


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("chelsia", "Chelsea FC"),
        ("bayer leverkusin", "Bayer 04 Leverkusen"),
        ("barca", "FC Barcelona"),
        ("psg", "Paris Saint-Germain FC"),
        ("inter", "FC Internazionale Milano"),
        ("man city", "Manchester City FC"),
        ("man utd", "Manchester United FC"),
        ("Arsenal", "Arsenal FC"),
        ("  MAN  CITY ", "Manchester City FC"),
        ("atleti", "Club Atlético de Madrid"),
        ("brighton", "Brighton & Hove Albion FC"),
        ("como", "Como 1907"),
    ],
)
def test_resolves(resolver, q, expected):
    assert resolver.resolve_query(q).name == expected


def test_user_aliases_resolve_names_fuzzy_matching_cannot(aliases):
    teams = [*TEAMS, Team(id=109, name="Juventus FC", short_name="Juventus", tla="JUV")]
    teams.append(Team(id=73, name="Tottenham Hotspur FC", short_name="Tottenham", tla="TOT"))
    resolver = TeamResolver(teams, aliases)
    assert resolver.resolve_query("juve").name == "Juventus FC"
    assert resolver.resolve_query("spurs").name == "Tottenham Hotspur FC"
    assert resolver.resolve_query("inter milan").name == "FC Internazionale Milano"


def test_a_low_confidence_single_match_is_offered_as_the_only_candidate():
    with pytest.raises(AmbiguousTeam) as e:
        TeamResolver(TEAMS, Aliases()).resolve_query("man utd")
    assert [t.name for t in e.value.candidates] == ["Manchester United FC"]


def test_teams_listed_under_several_competitions_are_one_team():
    resolver = TeamResolver([*TEAMS, *TEAMS], Aliases())
    assert resolver.resolve_query("arsenal").id == ARSENAL_ID
    assert resolver.map_source_names("understat", ["Arsenal"]) == {"Arsenal": ARSENAL_ID}


def test_real_is_ambiguous(resolver):
    with pytest.raises(AmbiguousTeam) as e:
        resolver.resolve_query("real")
    assert e.value.text == "real"
    assert {t.name for t in e.value.candidates} == {"Real Madrid CF", "Real Betis Balompié", "Real Sociedad de Fútbol"}


def test_milan_is_ambiguous(resolver):
    with pytest.raises(AmbiguousTeam) as e:
        resolver.resolve_query("milan")
    assert {t.name for t in e.value.candidates} == {"AC Milan", "FC Internazionale Milano"}


def test_alias_candidates_are_ordered_by_name(resolver):
    with pytest.raises(AmbiguousTeam) as e:
        resolver.resolve_query("real")
    assert [t.name for t in e.value.candidates] == ["Real Betis Balompié", "Real Madrid CF", "Real Sociedad de Fútbol"]


def test_fuzzy_ambiguity_orders_by_score_then_name():
    resolver = TeamResolver(TEAMS, Aliases())
    with pytest.raises(AmbiguousTeam) as e:
        resolver.resolve_query("man")
    assert [t.name for t in e.value.candidates] == ["Manchester City FC", "Manchester United FC"]


def test_fuzzy_real_without_alias_is_still_ambiguous():
    resolver = TeamResolver(TEAMS, Aliases())
    with pytest.raises(AmbiguousTeam) as e:
        resolver.resolve_query("real")
    assert {t.name for t in e.value.candidates} == {"Real Madrid CF", "Real Betis Balompié", "Real Sociedad de Fútbol"}


def test_alias_whose_teams_are_absent_falls_back_to_fuzzy():
    resolver = TeamResolver([t for t in TEAMS if t.name != "Real Madrid CF"], Aliases({"madrid": ("Real Madrid CF",)}))
    assert resolver.resolve_query("madrid").name == "Club Atlético de Madrid"


def test_alias_with_one_team_present_resolves_directly():
    teams = [t for t in TEAMS if t.name != "Real Betis Balompié"]
    aliases = Aliases({"rm": ("Real Madrid CF", "Real Betis Balompié")})
    assert TeamResolver(teams, aliases).resolve_query("rm").name == "Real Madrid CF"


def test_candidates_and_suggestions_do_not_depend_on_team_order():
    forward = TeamResolver(TEAMS, Aliases())
    backward = TeamResolver(list(reversed(TEAMS)), Aliases())
    with pytest.raises(AmbiguousTeam) as a:
        forward.resolve_query("man")
    with pytest.raises(AmbiguousTeam) as b:
        backward.resolve_query("man")
    assert a.value.candidates == b.value.candidates
    with pytest.raises(TeamNotFound) as c:
        forward.resolve_query("xyzzy")
    with pytest.raises(TeamNotFound) as d:
        backward.resolve_query("xyzzy")
    assert c.value.suggestions == d.value.suggestions


def _tla_trap_resolver() -> TeamResolver:
    teams = [
        *(t for t in TEAMS if t.name in {"Arsenal FC", "Chelsea FC", "Manchester City FC"}),
        Team(id=95, name="Valencia CF", short_name="Valencia", tla="VAL"),
        Team(id=328, name="Burnley FC", short_name="Burnley", tla="BUR"),
    ]
    return TeamResolver(teams, Aliases())


@pytest.mark.parametrize("query", ["valladolid", "burton albion"])
def test_a_tla_substring_never_confidently_resolves_to_the_wrong_team(query):
    resolver = _tla_trap_resolver()
    with pytest.raises((TeamNotFound, AmbiguousTeam)) as e:
        resolver.resolve_query(query)
    if isinstance(e.value, AmbiguousTeam):
        assert len(e.value.candidates) >= 1  # offered for confirmation, not accepted silently


@pytest.mark.parametrize(
    ("query", "expected"), [("val", "Valencia CF"), ("BUR", "Burnley FC"), ("mci", "Manchester City FC")]
)
def test_exact_tla_still_resolves(query, expected):
    assert _tla_trap_resolver().resolve_query(query).name == expected


def test_unknown(resolver):
    with pytest.raises(TeamNotFound) as e:
        resolver.resolve_query("xyzzy")
    assert e.value.text == "xyzzy"
    assert len(e.value.suggestions) == 3
    assert all(s in ID for s in e.value.suggestions)


@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("Arsenal v Chelsea", ("Arsenal", "Chelsea")),
        ("arsenal vs chelsea", ("arsenal", "chelsea")),
        ("arsenal VS. chelsea", ("arsenal", "chelsea")),
        ("Arsenal V Chelsea", ("Arsenal", "Chelsea")),
        ("Arsenal v. Chelsea", ("Arsenal", "Chelsea")),
        ("Arsenal – Chelsea", ("Arsenal", "Chelsea")),
        ("Arsenal — Chelsea", ("Arsenal", "Chelsea")),
        ("Paris Saint-Germain – Lyon", ("Paris Saint-Germain", "Lyon")),
        ("Arsenal - Chelsea", ("Arsenal", "Chelsea")),
        ("Brighton & Hove Albion v Real Betis", ("Brighton & Hove Albion", "Real Betis")),
        ("Paris Saint-Germain v Lyon", ("Paris Saint-Germain", "Lyon")),
        ("arsenal", ("arsenal", None)),
        ("  arsenal  ", ("arsenal", None)),
        ("Paris Saint-Germain", ("Paris Saint-Germain", None)),
    ],
)
def test_parse_query(q, expected):
    assert parse_query(q) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Real Sociedad de Fútbol", "real sociedad futbol"),
        ("FC Bayern München", "bayern munchen"),
        ("Club Atlético de Madrid", "atletico madrid"),
        ("Paris Saint-Germain FC", "paris saint germain"),
        ("Brighton & Hove Albion FC", "brighton hove albion"),
        ("AS Roma", "roma"),
        ("AC Milan", "milan"),
        ("FC København", "kobenhavn"),
        ("Bayer 04 Leverkusen", "bayer 04 leverkusen"),
        ("  Como   1907 ", "como 1907"),
        ("1. FC Köln", "koln"),
    ],
)
def test_normalize(name, expected):
    assert normalize(name) == expected


def test_normalize_keeps_a_name_made_only_of_club_form_tokens():
    assert normalize("AC") == "ac"


def test_source_names_unmatched_is_none(resolver):
    m = resolver.map_source_names("understat", ["Manchester City", "Arsenal", "Ipswich"])
    assert m["Ipswich"] is None
    assert m["Arsenal"] == ARSENAL_ID
    assert m["Manchester City"] == ID["Manchester City FC"]


def test_source_override_wins_over_fuzzy(resolver):
    m = resolver.map_source_names("clubelo", ["Inter", "Milan", "Man City"])
    assert m == {
        "Inter": ID["FC Internazionale Milano"],
        "Milan": ID["AC Milan"],
        "Man City": ID["Manchester City FC"],
    }


def test_source_override_for_a_team_outside_the_list_is_none():
    resolver = TeamResolver([t for t in TEAMS if t.name != "AC Milan"], Aliases.load())
    assert resolver.map_source_names("clubelo", ["Milan"]) == {"Milan": None}


def test_two_source_names_for_one_team_raises(resolver):
    with pytest.raises(SourceError) as e:
        resolver.map_source_names("understat", ["Manchester City", "Man City FC"])
    message = str(e.value)
    assert message.startswith("understat: ")
    assert "Manchester City" in message and "Man City FC" in message


def test_find_in_source_prefers_override(resolver):
    inter = next(t for t in TEAMS if t.name == "FC Internazionale Milano")
    assert resolver.find_in_source("clubelo", inter, ["Milan", "Inter", "Roma"]) == "Inter"


def test_find_in_source_falls_back_to_fuzzy(resolver):
    arsenal = next(t for t in TEAMS if t.id == ARSENAL_ID)
    assert resolver.find_in_source("understat", arsenal, ["Chelsea", "Arsenal", "Roma"]) == "Arsenal"


def test_find_in_source_without_entry_raises(resolver):
    arsenal = next(t for t in TEAMS if t.id == ARSENAL_ID)
    with pytest.raises(SourceError, match="understat: no entry for Arsenal FC"):
        resolver.find_in_source("understat", arsenal, ["Chelsea", "Roma"])


def test_packaged_aliases_have_the_required_entries(aliases):
    expected_direct = {
        "barca": "FC Barcelona",
        "inter": "FC Internazionale Milano",
        "psg": "Paris Saint-Germain FC",
        "man city": "Manchester City FC",
        "man utd": "Manchester United FC",
        "atleti": "Club Atlético de Madrid",
        "bayern": "FC Bayern München",
        "leverkusen": "Bayer 04 Leverkusen",
        "juve": "Juventus FC",
    }
    for alias, canonical in expected_direct.items():
        assert aliases.user[alias] == (canonical,)
    assert set(aliases.user["real"]) == {"Real Madrid CF", "Real Betis Balompié", "Real Sociedad de Fútbol"}
    assert set(aliases.user["milan"]) == {"AC Milan", "FC Internazionale Milano"}
    for section in ("understat", "clubelo", "odds_api", "fpl", "football_data_couk"):
        assert section in aliases.source
    assert aliases.source_override("clubelo", "Milan") == "AC Milan"
    assert aliases.source_override("clubelo", "Inter") == "FC Internazionale Milano"


def test_aliases_load_from_a_custom_path_normalises_keys(tmp_path):
    path = tmp_path / "aliases.toml"
    path.write_text(
        '[user]\n"Man Utd" = ["Manchester United FC"]\n[source.fpl]\n"Nott\'m Forest" = "Nottingham Forest FC"\n',
        encoding="utf-8",
    )
    loaded = Aliases.load(path)
    assert loaded.user == {"man utd": ("Manchester United FC",)}
    assert loaded.source_override("fpl", "NOTT'M forest") == "Nottingham Forest FC"
    assert loaded.source_override("fpl", "Arsenal") is None
    assert loaded.source_override("unknown_source", "Arsenal") is None
