"""Task identity from the instruction: two reads of one Qwen2.5-0.5B snapshot, multiplied.

A linear softmax head over centred embeddings is fit on each task's bank (every phrasing its demos
record, plus lossy surrogates), decay and temperature chosen by held-out likelihood; the snapshot
is also read generatively over the task names. The posterior is the renormalised product: the head
knows the corpus' wording, the generative read knows the task name's words. Soft, never a mask."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Iterable, Mapping, Sequence

import numpy as np

from robot_ood.core import npz

__all__ = [
    "LANG_NPZ",
    "QUERIES_NPZ",
    "QUERIES_ENV_VAR",
    "EMBED_DIM",
    "N_AUGMENT",
    "N_BANK",
    "AUGMENT_P_DROP",
    "augment_instruction",
    "normalise_instruction",
    "InstructionEncoder",
    "TaskEmbeddings",
]

LANG_NPZ: Final[str] = "g_lang.npz"

# Optional, beside LANG_NPZ: task posteriors of instructions embedded ahead of time, so a deploy
# box needs no encoder. Env var 0 ignores the file and embeds live.
QUERIES_NPZ: Final[str] = "g_lang_queries.npz"
QUERIES_ENV_VAR: Final[str] = "ROBOT_OOD_LANG_QUERIES"

# What to tell a caller whose g_lang.npz is missing.
MISSING_LANG: Final[str] = (
    "no task embeddings. Build them with `python -m robot_ood.cli build language --suite <suite>`."
)

# The local snapshot. Resolved through the env var first so a different box can point elsewhere
# without a code change; there is no download path -- HF_HUB_OFFLINE is forced on below.
MODEL_ENV_VAR: Final[str] = "ROBOT_OOD_LANG_MODEL"
_HUB_SUBDIR: Final[str] = "models--Qwen--Qwen2.5-0.5B/snapshots"


def _default_model_dir() -> str:
    """The Qwen2.5-0.5B snapshot parent in the HuggingFace cache.

    Returns:
        The first candidate that exists, or the default cache path.
    """
    candidates = [
        os.environ.get("HF_HUB_CACHE"),
        os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else None,
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    roots = [os.path.join(c, _HUB_SUBDIR) for c in candidates if c]
    return next((r for r in roots if os.path.isdir(r)), roots[-1])

EMBED_DIM: Final[int] = 896          # Qwen2.5-0.5B hidden size

_SCENE_PREFIX: Final[re.Pattern[str]] = re.compile(r"^[A-Z0-9_]*?SCENE\d+_")

_PERTURB_TAIL: Final[re.Pattern[str]] = re.compile(
    r"\s+(?:view|initstate|noise|language|tb)(?:\s+[-+0-9.]+)+\s*$"
)

N_AUGMENT: Final[int] = 16
AUGMENT_P_DROP: Final[float] = 0.3
N_BANK: Final[int] = 1024
HEAD_DECAYS: Final[tuple[float, ...]] = (0.0, 1e-6, 1e-5, 1e-4)
HEAD_HELD_OUT: Final[int] = 5
HEAD_MAX_ITER: Final[int] = 500


def augment_instruction(text: str, rng: np.random.Generator, p_drop: float = AUGMENT_P_DROP) -> str:
    """A lossy restatement of one instruction: drop words, then lightly reorder.

    Args:
        text: The instruction.
        rng: Random state.
        p_drop: Per-word drop probability. Never drops every word.

    Returns:
        The surrogate. Returns the normalised original when it has fewer than three words, where
        dropping anything would leave no content at all.
    """
    words = normalise_instruction(text).split()
    if len(words) < 3:
        return " ".join(words)
    keep = [w for w in words if rng.random() >= p_drop]
    if len(keep) < 2:
        keep = list(rng.choice(words, size=2, replace=False))
    # A single adjacent swap: word order carries some meaning, so scrambling outright would make
    # the surrogate harder than any real rewording rather than merely lossier.
    if len(keep) > 2 and rng.random() < 0.5:
        i = int(rng.integers(0, len(keep) - 1))
        keep[i], keep[i + 1] = keep[i + 1], keep[i]
    return " ".join(keep)


def normalise_instruction(text: str) -> str:
    """Normalize a task name by stripping scene prefix, perturbation tail, and underscores.

    Args:
        text: A graph task name or harness instruction string.

    Returns:
        The plain-language form, lowercased and whitespace-collapsed.
    """
    plain = _SCENE_PREFIX.sub("", str(text)).replace("_", " ")
    while True:
        trimmed = _PERTURB_TAIL.sub("", plain)
        if trimmed == plain:
            return " ".join(trimmed.lower().split())
        plain = trimmed


@dataclass(eq=False)
class InstructionEncoder:
    """Frozen Qwen2.5-0.5B, mean-pooled over the attention mask, memoised per string.

    Attributes:
        model_dir: Snapshot directory, or the parent holding exactly one snapshot.
        device: Torch device string. CPU is the default and is the right one: this runs once per
            episode on a string of a dozen tokens, and the GPU is holding the policy.
    """

    model_dir: str = ""
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._tok = None
        self._model = None
        self._lm = None
        self._cache: dict[str, np.ndarray] = {}
        self._ll_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

    # ---- resolution ------------------------------------------------------------------------------
    @property
    def snapshot(self) -> Path:
        """The concrete snapshot directory.

        Returns:
            The directory holding config.json and the tokenizer.

        Raises:
            FileNotFoundError: Nothing usable is at the resolved path. There is no download
                fallback: a run that silently fetched a different encoder than the one the artifact
                was built with would score a graph against embeddings from another model.
        """
        root = Path(self.model_dir or os.environ.get(MODEL_ENV_VAR) or _default_model_dir())
        if (root / "config.json").is_file():
            return root
        if root.is_dir():
            snaps = sorted(p for p in root.iterdir() if (p / "config.json").is_file())
            if len(snaps) == 1:
                return snaps[0]
            if len(snaps) > 1:
                raise FileNotFoundError(
                    f"{root}: {len(snaps)} snapshots, cannot choose. Point {MODEL_ENV_VAR} at one."
                )
        raise FileNotFoundError(
            f"{root}: no instruction encoder. This is the local Qwen2.5-0.5B snapshot; set "
            f"{MODEL_ENV_VAR} to its path. Nothing here downloads -- an encoder that differs from "
            f"the one {LANG_NPZ} was built with would produce embeddings the graph cannot be "
            f"compared against."
        )

    @property
    def model_id(self) -> str:
        """A stable identifier stamped into the artifact, so a mismatch is detectable."""
        return self.snapshot.name

    def _load(self):
        """Load the tokenizer and model once, offline.

        Returns:
            (tokenizer, model) with the model in eval mode.
        """
        if self._model is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import torch
            from transformers import AutoModel, AutoTokenizer

            snap = os.fspath(self.snapshot)
            tok = AutoTokenizer.from_pretrained(snap)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self._tok = tok
            self._model = AutoModel.from_pretrained(snap, torch_dtype=torch.float32).eval().to(
                self.device
            )
        return self._tok, self._model

    # ---- encoding --------------------------------------------------------------------------------
    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Embed instructions, mean-pooled over real tokens.

        Args:
            texts: Instruction strings; normalised here, so callers may pass raw task names.
            batch_size: Rows per forward pass.

        Returns:
            [n, EMBED_DIM] float32, unnormalised and uncentred. Centering is TaskEmbeddings' job
            because it needs the corpus mean, which one string does not have.

        Raises:
            ValueError: texts is empty.
        """
        if len(texts) == 0:
            raise ValueError("nothing to encode")
        import torch

        tok, model = self._load()
        plain = [normalise_instruction(t) for t in texts]
        missing = [p for p in dict.fromkeys(plain) if p not in self._cache]
        for i in range(0, len(missing), batch_size):
            chunk = missing[i : i + batch_size]
            batch = tok(chunk, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                hidden = model(**batch).last_hidden_state              # [b, T, EMBED_DIM]
            mask = batch["attention_mask"][..., None].to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            for text, row in zip(chunk, pooled.cpu().numpy().astype(np.float32)):
                self._cache[text] = row
        return np.stack([self._cache[p] for p in plain])

    def _load_lm(self):
        """The same snapshot with its language-model head, loaded once, offline."""
        if self._lm is None:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            snap = os.fspath(self.snapshot)
            if self._tok is None:
                tok = AutoTokenizer.from_pretrained(snap)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                self._tok = tok
            self._lm = AutoModelForCausalLM.from_pretrained(
                snap, torch_dtype=torch.float32
            ).eval().to(self.device)
            # One copy of the weights: the pooled encoder reads the trunk inside the LM.
            if self._model is None:
                self._model = self._lm.model
        return self._tok, self._lm

    def task_log_likelihood(self, text: str, task_texts: Sequence[str]) -> np.ndarray:
        """log P(task text | instruction) under the snapshot, one entry per task.

        Args:
            text: The live instruction.
            task_texts: The canonical task descriptions, plain words.

        Returns:
            [n_tasks] float64 per-token mean log-probability of each task text after the prompt,
            rescaled by the mean task length so the entries compare like whole-text likelihoods
            without favouring short names.
        """
        key = (normalise_instruction(text), tuple(task_texts))
        if key in self._ll_cache:
            return self._ll_cache[key]
        import torch

        tok, lm = self._load_lm()
        head = f"Instruction: {key[0]}\nThe robot task is:"
        prompts = [f"{head} {t}" for t in task_texts]
        n_head = len(tok(head)["input_ids"])
        batch = tok(prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            logits = lm(**batch).logits[:, :-1]
        target = batch["input_ids"][:, 1:]
        lp = torch.log_softmax(logits, dim=-1).gather(-1, target[..., None])[..., 0]
        out = np.zeros(len(task_texts), dtype=np.float64)
        n_tok = np.zeros(len(task_texts), dtype=np.float64)
        for j in range(len(task_texts)):
            length = int(batch["attention_mask"][j].sum().item())
            span = lp[j, n_head - 1 : length - 1]
            out[j] = float(span.mean().item())
            n_tok[j] = float(span.numel())
        out = out * n_tok.mean()
        self._ll_cache[key] = out
        return out

    def encode_one(self, text: str) -> np.ndarray:
        """Embed a single instruction.

        Args:
            text: The instruction.

        Returns:
            [EMBED_DIM] float32.
        """
        return self.encode([text])[0]


def _fit_head(x: np.ndarray, y: np.ndarray, n_tasks: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    """A linear softmax head over unit embeddings, its weight decay and temperature chosen on held-out rows.

    Args:
        x: [n, dim] float32 rows in the table's centred unit space.
        y: [n] task id per row.
        n_tasks: Tasks in the graph.
        seed: Random state for the held-out split.

    Returns:
        weight [dim, n_tasks], bias [n_tasks] refit on every row under the chosen decay, and the
        temperature that minimises the held-out negative log-likelihood of that decay's held-out fit.
    """
    import torch

    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.int64)
    held = torch.zeros(len(y), dtype=torch.bool)
    held[torch.as_tensor(np.random.default_rng(seed).permutation(len(y))[: len(y) // HEAD_HELD_OUT])] = True
    if held.sum() == 0 or (~held).sum() == 0:
        held[:] = False

    def fit(rows: torch.Tensor, labels: torch.Tensor, decay: float) -> tuple[torch.Tensor, torch.Tensor]:
        w = torch.zeros(rows.shape[1], n_tasks, requires_grad=True)
        b = torch.zeros(n_tasks, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=HEAD_MAX_ITER, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(rows @ w + b, labels) + decay * (w ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        return w.detach(), b.detach()

    best: tuple[float, float, torch.Tensor] | None = None
    for decay in HEAD_DECAYS:
        w, b = fit(xt[~held], yt[~held], decay)
        logits = xt[held] @ w + b if held.any() else xt @ w + b
        labels = yt[held] if held.any() else yt
        nll = float(torch.nn.functional.cross_entropy(logits, labels))
        if best is None or nll < best[0]:
            best = (nll, decay, logits)
    _, decay, logits = best
    labels = yt[held] if held.any() else yt
    log_t = torch.zeros((), requires_grad=True)
    opt = torch.optim.LBFGS([log_t], max_iter=HEAD_MAX_ITER)

    def closure_t() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure_t)
    w, b = fit(xt, yt, decay)
    return w.numpy().astype(np.float32), b.numpy().astype(np.float32), float(log_t.exp())


@dataclass(frozen=True, eq=False)
class TaskEmbeddings:
    """The graph's per-task instruction table: centred embeddings, the phrasing bank, and the head.

    Attributes:
        e_task: [n_tasks, EMBED_DIM] float32 bank centroids, centred and L2-normalised.
        mean: [EMBED_DIM] float32 mean of the raw centroids -- what a query must be centred by to
            live in the same space. Shipped rather than recomputed: a deployed episode sees one
            instruction and cannot estimate a corpus mean.
        model_id: The snapshot the embeddings came from.
        task_names: The raw task names, in task-id order.
        e_aug: [n_bank, EMBED_DIM] float32 rows of the phrasing bank, centred by the same mean;
            task t owns rows aug_ptr[t]:aug_ptr[t + 1]. Training draws its paraphrases from these.
        aug_ptr: [n_tasks + 1] int64 row offsets into e_aug.
        head_w: [EMBED_DIM, n_tasks] float32 head weights.
        head_b: [n_tasks] float32 head bias.
        temperature: The head's logit divisor.
        p_task: [n_tasks, n_tasks] float32 head posterior of each centroid.
        p_aug: [n_bank, n_tasks] float32 head posterior of each bank row.
        queries: Normalised instruction -> [n_tasks] float64 task_posterior, embedded ahead of
            time by write_queries. task_posterior reads here before it reads the encoder.
    """

    e_task: np.ndarray
    mean: np.ndarray
    model_id: str
    task_names: tuple[str, ...]
    e_aug: np.ndarray
    aug_ptr: np.ndarray
    head_w: np.ndarray
    head_b: np.ndarray
    temperature: float
    p_task: np.ndarray
    p_aug: np.ndarray
    queries: Mapping[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_tasks(self) -> int:
        """Number of tasks in the graph."""
        return int(self.e_task.shape[0])

    @staticmethod
    def _unit(rows: np.ndarray) -> np.ndarray:
        """L2-normalise along the last axis, leaving a zero row zero.

        Args:
            rows: [..., d] float array.

        Returns:
            [..., d] float32.
        """
        norm = np.linalg.norm(rows, axis=-1, keepdims=True)
        return (rows / np.where(norm > 1e-12, norm, 1.0)).astype(np.float32)

    @classmethod
    def build(
        cls, task_names: Iterable[str], encoder: InstructionEncoder,
        phrasings: Sequence[Sequence[str]] | None = None,
        n_augment: int = N_AUGMENT, n_bank: int = N_BANK, seed: int = 0,
    ) -> "TaskEmbeddings":
        """Encode each task's phrasing bank, centre the table on the centroids, and fit the head.

        Args:
            task_names: The graph's task names, in task-id order.
            encoder: The instruction encoder.
            phrasings: Per task, the phrasings its demonstrations record; None reads each task
                name as its only phrasing.
            n_augment: Lossy surrogates per task, drawn from its phrasings; 0 builds none.
            n_bank: Bank rows kept per task in e_aug; the head is fit on every row regardless.
            seed: Random state for the surrogates, the bank sample and the head's held-out split.

        Returns:
            The embeddings.

        Raises:
            ValueError: phrasings does not hold one non-empty sequence per task.

        The mean that centres everything is the mean of the bank CENTROIDS alone, so two graphs
        built with different n_augment place the same instruction at the same coordinates.
        """
        names = tuple(str(n) for n in task_names)
        recorded = [[str(t) for t in per] for per in (phrasings if phrasings is not None else [[n] for n in names])]
        if len(recorded) != len(names) or any(not per for per in recorded):
            raise ValueError(f"expected one non-empty phrasing list per task, {len(names)} tasks")
        rng = np.random.default_rng(seed)
        banks = [
            per + [augment_instruction(per[int(rng.integers(len(per)))], rng) for _ in range(n_augment)]
            for per in recorded
        ]
        owner = np.concatenate([np.full(len(b), i, dtype=np.int64) for i, b in enumerate(banks)])
        raw = encoder.encode([t for b in banks for t in b])
        centroids = np.stack([raw[owner == i].mean(axis=0) for i in range(len(names))])
        mean = centroids.mean(axis=0)
        rows = cls._unit(raw - mean)
        head_w, head_b, temperature = _fit_head(rows, owner, len(names), seed)

        kept = np.concatenate([
            np.sort(rng.choice(np.flatnonzero(owner == i), min(n_bank, len(banks[i])), replace=False))
            for i in range(len(names))
        ])
        aug_ptr = np.concatenate([[0], np.cumsum([min(n_bank, len(b)) for b in banks])]).astype(np.int64)
        table = cls(
            e_task=cls._unit(centroids - mean), mean=mean.astype(np.float32),
            model_id=encoder.model_id, task_names=names, e_aug=rows[kept], aug_ptr=aug_ptr,
            head_w=head_w, head_b=head_b, temperature=temperature,
            p_task=np.zeros((0, 0), dtype=np.float32), p_aug=np.zeros((0, 0), dtype=np.float32),
        )
        texts = [t for b in banks for t in b]
        p_aug = np.stack([table.posterior(rows[k], table.names_posterior(texts[k], encoder)) for k in kept])
        p_task = np.stack([p_aug[aug_ptr[i]:aug_ptr[i + 1]].mean(axis=0) for i in range(len(names))])
        return cls(**{**table.__dict__, "p_task": p_task.astype(np.float32), "p_aug": p_aug.astype(np.float32)})

    def posterior(self, rows: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
        """P(task | row) for rows already in the table's centred unit space: the head, times a prior.

        Args:
            rows: [..., EMBED_DIM] float32.
            prior: [..., n_tasks] to multiply in and renormalise, or None for the head alone.

        Returns:
            [..., n_tasks] float32 summing to one over the last axis.
        """
        z = (np.asarray(rows, dtype=np.float32) @ self.head_w + self.head_b) / self.temperature
        z = np.exp(z - z.max(axis=-1, keepdims=True))
        if prior is not None:
            z = z * np.asarray(prior, dtype=np.float64)
        return (z / z.sum(axis=-1, keepdims=True)).astype(np.float32)

    def names_posterior(self, text: str, encoder: InstructionEncoder) -> np.ndarray:
        """P(task | instruction) from the snapshot read generatively over the task names, uniform prior.

        The read is the pointwise mutual information between the instruction and each task name:
        log P(name | instruction) minus log P(name | no instruction). The snapshot has its own
        preference among the names before it sees any instruction (on RoboTwin's terse names it
        puts "handover block" 10-14 nats above "blocks ranking rgb"), and that preference is not
        evidence about this instruction; left in, it reads most paraphrases as the favoured name and,
        multiplied into the head, overrides a head that had them right. Every corpus instruction
        still reads as its own task, because the instruction repeats the name.
        ROBOT_OOD_LANG_POSTERIOR=raw keeps the uncalibrated read.

        Args:
            text: An instruction.
            encoder: The same encoder the artifact was built with.

        Returns:
            [n_tasks] float64 summing to one.
        """
        names = [normalise_instruction(n) for n in self.task_names]
        ll = encoder.task_log_likelihood(text, names)
        if os.environ.get("ROBOT_OOD_LANG_POSTERIOR") != "raw":
            ll = ll - encoder.task_log_likelihood("", names)
        p = np.exp(ll - ll.max())
        return p / p.sum()

    def draw_aug(self, task: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Bank row indices: for each task one row picked uniformly from its own rows.

        Args:
            task: [n] task ids.
            u: [n] uniform draws in [0, 1).

        Returns:
            [n] int64 indices into e_aug.
        """
        lo, hi = self.aug_ptr[task], self.aug_ptr[np.asarray(task) + 1]
        return lo + np.minimum((np.asarray(u) * (hi - lo)).astype(np.int64), hi - lo - 1)

    def embed_query(self, text: str, encoder: InstructionEncoder) -> np.ndarray:
        """Encode one live instruction into the task embeddings' own space.

        Args:
            text: The harness's instruction string for this episode.
            encoder: The same encoder the artifact was built with.

        Returns:
            [EMBED_DIM] float32, centred and L2-normalised.
        """
        return self._unit(encoder.encode_one(text) - self.mean)

    def task_posterior(self, text: str, encoder: InstructionEncoder) -> np.ndarray:
        """P(task | instruction): the head's read times the generative read, renormalised.

        Args:
            text: The harness's instruction string for this episode.
            encoder: The same encoder the artifact was built with.

        Returns:
            [n_tasks] float64 summing to one.
        """
        known = self.queries.get(normalise_instruction(text))
        if known is not None:
            return np.asarray(known, dtype=np.float64)
        # ROBOT_OOD_LANG_POSTERIOR=head drops the generative read over the task names altogether.
        if os.environ.get("ROBOT_OOD_LANG_POSTERIOR") == "head":
            return self.posterior(self.embed_query(text, encoder)).astype(np.float64)
        return self.posterior(self.embed_query(text, encoder),
                              self.names_posterior(text, encoder)).astype(np.float64)

    def write_queries(self, path: str | os.PathLike[str], texts: Sequence[str],
                      encoder: InstructionEncoder) -> Path:
        """Embed instructions now and store their task posterior beside the graph artifacts.

        Args:
            path: Destination .npz, or the directory to write QUERIES_NPZ into.
            texts: The instructions a deploy will send, verbatim.
            encoder: The same encoder the artifact was built with.

        Returns:
            The written path.
        """
        plain = list(dict.fromkeys(normalise_instruction(t) for t in texts))
        return npz.write_arrays(path, QUERIES_NPZ, {
            "model_id": np.array(self.model_id),
            "texts": np.array(plain),
            "p_task": np.stack([self.task_posterior(t, encoder) for t in plain]),
        })

    def embed_query_posterior(self, text: str, encoder: InstructionEncoder) -> np.ndarray:
        """The canonical task embeddings mixed by task_posterior.

        Args:
            text: The harness's instruction string for this episode.
            encoder: The same encoder the artifact was built with.

        Returns:
            [EMBED_DIM] float32, in the table's own centred unit space.
        """
        return self._unit(self.task_posterior(text, encoder).astype(np.float32) @ self.e_task)

    # ---- persistence -----------------------------------------------------------------------------
    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the table beside the graph artifacts.

        Args:
            path: Destination .npz, or the directory to write LANG_NPZ into.

        Returns:
            The written path.
        """
        return npz.write_arrays(path, LANG_NPZ, {
            "e_task": self.e_task, "mean": self.mean,
            "model_id": np.array(self.model_id), "task_names": np.array(self.task_names),
            "e_aug": self.e_aug, "aug_ptr": self.aug_ptr, "head_w": self.head_w,
            "head_b": self.head_b, "temperature": np.array(self.temperature, dtype=np.float32),
            "p_task": self.p_task, "p_aug": self.p_aug,
        })

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "TaskEmbeddings":
        """Read a table written by save.

        Args:
            path: The .npz, or the directory holding LANG_NPZ.

        Returns:
            The table.

        Raises:
            FileNotFoundError: Nothing is at the resolved path.
            ValueError: The archive predates the head.
        """
        src, archive = npz.read(path, LANG_NPZ, MISSING_LANG)
        if "head_w" not in archive:
            raise ValueError(f"{LANG_NPZ} carries no head; rebuild it with `build language`")
        queries: dict[str, np.ndarray] = {}
        if os.environ.get(QUERIES_ENV_VAR, "1") != "0" and (src.parent / QUERIES_NPZ).is_file():
            _, table = npz.read(src.parent, QUERIES_NPZ, "")
            if str(table["model_id"]) == str(archive["model_id"]):
                queries = {str(t): np.asarray(p, dtype=np.float64)
                           for t, p in zip(table["texts"], table["p_task"])}
        return cls(
            e_task=np.asarray(archive["e_task"], dtype=np.float32),
            mean=np.asarray(archive["mean"], dtype=np.float32),
            model_id=str(archive["model_id"]),
            task_names=tuple(str(n) for n in archive["task_names"]),
            e_aug=np.asarray(archive["e_aug"], dtype=np.float32),
            aug_ptr=np.asarray(archive["aug_ptr"], dtype=np.int64),
            head_w=np.asarray(archive["head_w"], dtype=np.float32),
            head_b=np.asarray(archive["head_b"], dtype=np.float32),
            temperature=float(archive["temperature"]),
            p_task=np.asarray(archive["p_task"], dtype=np.float32),
            p_aug=np.asarray(archive["p_aug"], dtype=np.float32),
            queries=queries,
        )

    def check_against(self, task_names: Sequence[str]) -> "TaskEmbeddings":
        """Refuse a table built for a different graph.

        Args:
            task_names: The node table's task names, in task-id order.

        Returns:
            self, so the check can be chained onto a load.

        Raises:
            ValueError: The task count or the names disagree. A table silently indexed by a
                different task ordering would tilt retrieval towards the wrong demonstrations and
                still look well-formed.
        """
        theirs = tuple(str(n) for n in task_names)
        if theirs != self.task_names:
            raise ValueError(
                f"{LANG_NPZ} was built for {len(self.task_names)} tasks {self.task_names[:2]}... "
                f"but this graph has {len(theirs)} {theirs[:2]}.... Rebuild it."
            )
        return self
