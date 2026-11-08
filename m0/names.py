"""Node-name pool and token rules (CONFIG.md §1.1-1.2, item F-1).

POOL: 1,000 neutral, common English nouns. Words that could interact with the task are excluded:
color words and words strongly associated with a color (COLOR_BLACKLIST), answer words, graph vocabulary
and template words (TASK_BLACKLIST).

Token rule: every occurrence of a name in a prompt is preceded by a single space (serialize.py writes every
line with a leading space), so a name always appears as the token " w". A name is kept only if " w" is a
single token for every tokenizer under test (single_token_names), and every prompt is checked with
verify_name_tokens: all occurrences of each name must map to one and the same token id.
"""
import json
from pathlib import Path

COLOR_BLACKLIST = set("""
red blue green yellow orange purple pink brown black white gray grey violet indigo cyan magenta
cobalt salmon ivory silver copper cherry tomato lemon ginger mango pumpkin olive plum peach amber coral
rose lime gold golden bronze crimson scarlet maroon navy azure teal turquoise lavender lilac beige tan
cream emerald ruby sapphire jade pearl charcoal ebony ash rust honey mustard chocolate caramel cocoa
coffee sand snow ink blood sky grass cherry berry banana apricot lemonade carrot strawberry raspberry
blueberry orchid tulip denim khaki mint sage moss pine slate steel graphite marble milk butter
""".split())
TASK_BLACKLIST = set("""
yes no node nodes edge edges graph neighbor neighbors color colors colored none is true false
question answer red blue green kit
key string data source target attr type name id for graphml undirected here given description using above
following single word reply respond only every each
""".split())   # task, format, GraphML markup and template vocabulary

_RAW = """
anchor arrow badge bakery banjo barrel basket beacon bench biscuit blanket bottle bracket bridge bubble
bucket button cabin cactus camel candle canyon carpet castle cereal chalk chimney circus cliff clock cloud
cookie cotton crayon cricket crystal curtain cushion dagger desert diamond dolphin donkey dragon drum eagle
easel elbow engine falcon feather fence ferry fiddle forest fossil fountain fox galaxy garden garlic giant
glacier globe goblet gravel guitar hammer harbor helmet hermit hollow hornet island jacket jaguar jelly
jungle kettle kingdom kitten ladder lantern library lizard lobster locket magnet maple meadow melon mirror
monkey mosaic muffin needle nickel noodle oasis oyster paddle palace panda parrot pebble pencil pepper piano
pickle pillow pirate planet pocket potato puzzle quarry quilt rabbit radar raven ribbon rocket saddle sandal
satin scarf shadow shovel slipper socket spider sponge squirrel statue stove sugar summit sunset tablet
teapot temple ticket tiger timber tower tractor trumpet tunnel turtle umbrella valley velvet violin volcano
wagon walnut walrus whistle willow window wizard yogurt zebra
acorn actor album alley altar anvil apron arch arena armor attic avenue axle bacon badger bagel balcony
ballot bamboo bandit banner barber barn baron basin bat beak beard beaver bed bee beetle bell belt beret
bicycle bishop blade blender blossom boat bolt bonnet book boot bottle boulder bow bowl box bracelet brick
bride broom brush bucket buffalo bugle bull bundle bunker burger bus bush butcher cable cafe cage cake
calendar camera camp canal candy cannon canoe cap captain car caravan card cargo carpet cart cartoon cave
cedar cell cellar chain chair chamber channel chapel charm chart chef chess chest chick chin chip choir
church cider cigar cinema citizen city clam clarinet classroom claw clay clerk clinic cloak closet clover
clown club coach coast coat cobra cocoon coin collar colony comb compass concert cone cook corn corner
cottage couch council counter county court cousin cow crab crane crate creek crew crib crocodile crow
crown crumb cube cucumber cup cupboard dancer dart deck deer den dentist desk dial diary dice dinner
dish doctor dock dog doll dome door dove dragon drawer dress drill drummer duck dune dust eel egg elephant
elk emperor envelope eraser factory fairy farm farmer faucet fern festival field fig file finger fish
flag flame flask fleet flute fly foam folder fork fort fridge frog frost fruit funnel fur gate gazelle
gear gem ghost gift giraffe glass glove goat goose gorilla gown grape grill guard gull gym hall
hamster hand hanger harp hat hawk hay hedge heel hen hill hippo hive hook horn horse hose hotel house
hut igloo iron jar jeep jet jewel judge jug kangaroo key king kiosk kite knee knife knight koala lab
lace ladle lake lamb lamp lane laptop latch lawn leaf ledge leopard letter lever lid lion lock lodge
log loom lotus lounge lynx mail mailbox mall mammoth map mask mast mat mayor maze medal menu mill
miner minnow mitten mole moose mop moth motor mouse mug mule museum nail napkin nest net newt notebook
nurse oak oar octopus office onion opera orbit organ ostrich otter oven owl ox pad pail pan pancake
panther parade park parcel pasta pastry path patio pea pear pedal pelican pen penguin pepper pier pig
pigeon pilot pin pipe pizza plank plate platform plaza plow plug plum poet pole pond pony pool porch
port post pot pouch prince printer prism pudding pump puppet puppy quail queen rack raft rail rake ram
ranch rat razor reef rhino ring river road robe robin robot rock roof room rooster root rope rug
ruler sack sail sailor salad salt sandwich sauce sausage saw scale school scissors scooter screen screw
seal seed shark sheep shelf shell shield ship shirt shoe shop shore shrimp sink skate skirt skull sled
sloth snail snake soap sock sofa soldier soup spade spoon spring stable stage stamp star station
steam stick stool store stork stream street string submarine suit swan sweater sword table tack tail
tank tape taxi tea teacher tent theater thimble thread throne thumb tile toad toaster toe tomb tongue
tool tooth torch towel town toy track trail train tray tree trolley truck trunk tub tuba tuna turkey
vase vault vest village vine wall wallet wand wasp watch wave well whale wheel wig wing wolf worm yacht
yak yard zipper zoo
admiral almond anthem antler apple archer atlas avocado bagpipe ballet bandage barley basil beach bead
bean bear bucket buckle bumper bunny cabbage canary candle cantor carnival carrot cashew catapult cathedral
cavern celery cello chariot cheetah chestnut chipmunk cinnamon citadel clamp cobbler cockpit comedian
condor coyote cradle crater cricket cupcake dandelion daisy dingo dolphin dragonfly dumpling eclipse
emu falcon ferret flamingo forge furnace gadget gallery gargoyle geyser gibbon glider gnome gondola
gopher grotto hammock harpoon hazel hedgehog heron hyena iguana jasmine javelin jester juggler kayak
kernel kiwi lagoon lasso lemur lighthouse llama lobby lollipop macaw magpie mandolin manor marmot
mermaid meteor minstrel moat monsoon mussel narwhal nectar nomad nutmeg oatmeal ocelot orchard oriole
paddock pagoda papaya parsley peacock pecan pendulum pepperoni petal pheasant piccolo pinecone pistachio
platypus pretzel puffin pyramid quiver raccoon radish raisin rattle reindeer relic rhubarb riddle
saffron sapling sardine scallop scepter scorpion seahorse sequoia sheriff sherpa skunk sleigh snorkel
sombrero sparrow spinach squash stallion starfish stingray sundial swallow tadpole tamarind tambourine
tapir tavern terrier thistle thrush toucan trellis trombone truffle tugboat turnip unicorn urchin
vulture waffle walrus warbler weasel wigwam windmill wombat woodpecker yodel zeppelin
abbey accordion aerial airship alcove ambulance amulet aquarium arcade armchair asteroid avalanche
backpack ballroom bandstand barge barometer bassoon bazaar bellows binoculars blimp boomerang bouquet
bowtie bridle brooch buggy bungalow cafeteria calculator canopy capsule carousel cassette cauldron
chandelier chapter chisel cleaver coaster cobweb colander comet compass corridor courtyard cowboy
crossbow cupola cutlass derrick dinghy dirigible dormitory drawbridge dustpan earmuff escalator
fireplace flagpole flashlight footbridge fortress freighter gazebo glacier granary greenhouse
hairbrush handcart harmonica hatchet haystack headlamp heliport hourglass houseboat hovercraft icicle
infirmary inkwell jukebox kennel keyboard keychain kimono ladle lamppost landmark lectern lifeboat
locomotive lunchbox mailbag mansion marina mattress megaphone microscope minaret monastery monocle
monument mortar motel mousetrap nightstand observatory obelisk orchestra paddleboat padlock palisade
pantry parachute parasol passport pavilion periscope pharmacy pinwheel pitchfork planetarium playground
pliers pocketknife podium postcard propeller pulley pushcart quarterback racetrack rafter rampart
rattlesnake recliner refinery rickshaw rowboat saddlebag sandbox satchel scaffold schooner scoreboard
seesaw sextant shipyard signpost skyscraper slingshot snowplow spaceship spatula speedboat spyglass
stadium stagecoach staircase stapler stethoscope stopwatch sunroof surfboard suitcase sundae tapestry
teacup teaspoon telescope thermos tiara toboggan tollbooth toolbox tram trampoline treehouse tricycle
trophy tugboat turbine turnstile typewriter ukulele vineyard wardrobe warehouse washboard watchtower
waterfall weathervane wheelbarrow whirlpool windsock workbench wristwatch xylophone
"""

POOL = []
for w in _RAW.split():
    if w in COLOR_BLACKLIST or w in TASK_BLACKLIST or w in POOL:
        continue
    POOL.append(w)
POOL = POOL[:1000]


# characters that can follow a name in a prompt (serializations, templates, GraphML attributes); a name passes only
# if " w" stays one and the same token id before every one of them (and at the end of the text)
CONTEXTS = [" ", "\n", ",", ":", ".", "?", ";", ")", '"', "'", "-", ""]
MASK_WORD = "kit"


def name_ok(tok, w):
    """True iff " w" is one token and that token id is kept in every following context (cheap: once per name and
    tokenizer, instead of tokenizing every prompt)."""
    t = tok.encode(" " + w, add_special_tokens=False)
    if len(t) != 1 or t[0] == getattr(tok, "unk_token_id", None):
        return False
    for c in CONTEXTS:
        ids = tok.encode("x " + w + c, add_special_tokens=False)
        if t[0] not in ids:
            return False
    return True


def single_token_names(tokenizers, pool=POOL, report=None, load=None):
    """Names w that pass name_ok for every tokenizer in `tokenizers` (HF model ids, or tokenizer objects with
    `load`); names that fail are dropped and logged (CONFIG §1.2). The mask word must pass as well (" kit"; "kit"
    without the space is reported). Writes {"pool", "kept", "per_tokenizer", "dropped", "mask_word", "names"}
    to `report`."""
    if load is None:
        from transformers import AutoTokenizer
        load = AutoTokenizer.from_pretrained
    keep, per, dropped, mask = list(pool), {}, {}, {}
    for name in tokenizers:
        tok = load(name)
        ok = {w for w in pool if name_ok(tok, w)}
        per[name] = len(ok)
        dropped[name] = [w for w in pool if w not in ok]
        keep = [w for w in keep if w in ok]
        mask[name] = {"with_space_ok": name_ok(tok, MASK_WORD),
                      "without_space_tokens": len(tok.encode(MASK_WORD, add_special_tokens=False))}
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"pool": len(pool), "kept": len(keep), "per_tokenizer": per, "dropped": dropped,
                   "mask_word": mask, "contexts": CONTEXTS, "names": keep}, open(report, "w"), indent=1)
    bad = [n for n, m in mask.items() if not m["with_space_ok"]]
    if bad:
        raise SystemExit(f"mask word ' {MASK_WORD}' is not one stable token for {bad}")
    return keep


def occurrences_ok(prompt, names):
    """Cheap per-prompt check (every prompt): each whole-word occurrence of a name is preceded by a space and
    followed by one of CONTEXTS (or the end); together with name_ok this fixes one token id per name."""
    import re
    for w in set(names):
        for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", prompt):
            a, b = m.start(), m.end()
            if a == 0 or prompt[a - 1] != " " or (b < len(prompt) and prompt[b] not in CONTEXTS):
                return False
    return True


def verify_name_tokens(tokenizer, prompt, names):
    """F-1 check. True iff every whole-word occurrence of every name w in `prompt` is preceded by a space and
    the span " w" is encoded as one and the same single token id. Returns (ok, details) with
    details = {name: set of observed token-id tuples (None for an occurrence without a leading space)}."""
    import re
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    details, ok = {}, True
    for w in set(names):
        seen = set()
        for m in re.finditer(r"(?<![A-Za-z0-9])" + re.escape(w) + r"(?![A-Za-z0-9])", prompt):
            a, b = m.start(), m.end()
            if a == 0 or prompt[a - 1] != " ":
                seen.add(None)
                continue
            a -= 1
            toks = tuple(i for i, (s, e) in zip(ids, offs) if e > a and s < b)
            # the token(s) must cover exactly " w"
            spans = [(s, e) for (s, e) in offs if e > a and s < b]
            # SentencePiece fast tokenizers (Gemma) may report the offset of '▁w' from the letter, not the space
            if spans and (spans[0][0] not in (a, a + 1) or spans[-1][1] != b):
                toks = toks + ("misaligned",)
            seen.add(toks)
        details[w] = seen
        if len(seen) != 1 or any(t is None or len(t) != 1 for t in seen):
            ok = False
    return ok, details


if __name__ == "__main__":
    print(len(POOL), "names in the pool")
