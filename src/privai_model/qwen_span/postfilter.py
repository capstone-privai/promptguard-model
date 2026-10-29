"""Drop predicted spans whose text is syntax that cannot itself be a credential value.

The generative model copies references (`${VAR}`, `{{ secrets.X }}`, `vault://...`), masks
(`****`, `[REDACTED]`), placeholders (`<YOUR_KEY>`, `changeme`) and lookup calls out of
the prompt as if they were secrets. Each rule matches the WHOLE span text, so a literal
value that merely contains `$` or `{` is kept. Rules are by syntax category, never by a
specific dataset string; the cost is measured by how many gold spans they would remove.
"""
import re

_I = re.IGNORECASE
RULES = {
    'mask_marker': re.compile(r'[*•xX#.\-]{3,}|[\[<(]?(redacted|filtered|hidden|masked|removed|omitted|secret)[\]>)]?', _I),
    'empty_or_null': re.compile(r'""|\'\'|null|none|nil|undefined', _I),
    'angle_placeholder': re.compile(r'<[^<>\s][^<>]*>'),
    'word_placeholder': re.compile(r'replace[-_ ]?me|change[-_ ]?me|todo|tbd|placeholder|your[-_ ][\w-]*|[\w-]*[-_ ]here', _I),
    'shell_variable': re.compile(r'\$\{?[A-Za-z_]\w*(:?[-=?+][^}]*)?\}?|%[A-Za-z_]\w*%|\$env:\w+|#\{\w+\}', _I),
    'command_substitution': re.compile(r'\$\(.+\)|`.+`', re.S),
    'template_expression': re.compile(r'\$?\{\{.+\}\}|<%=?.+%>|\{%.+%\}', re.S),
    # Calls only when every argument is quoted (optionally keyword=), so `Pa55(word)` stays a value.
    'lookup_call': re.compile(r'[A-Za-z_][\w.:]*\(\s*(\w+\s*=\s*)?["\'][^"\']*["\']'
                              r'(\s*,\s*(\w+\s*=\s*)?["\'][^"\']*["\'])*\s*\)\??;?'
                              r'|[\w.]+\[\s*["\'][^"\']+["\']\s*\]|process\.env\.\w+'),
    'secret_store_ref': re.compile(r'(vault|op|sm|gsm|awssm|azurekeyvault|secretsmanager|ssm)://\S+|arn:aws:(secretsmanager|ssm):\S+'
                                   r'|projects/[^/\s]+/secrets/\S+|@Microsoft\.KeyVault\(.+\)', _I),
}


def non_value_rule(value):
    """Name of the first rule the whole value matches, else None."""
    stripped = value.strip().strip('"\'') if len(value.strip()) > 2 else value.strip()
    for name, pattern in RULES.items():
        if pattern.fullmatch(value.strip()) or (stripped and pattern.fullmatch(stripped)):
            return name
    return None


# A lookup key passed to a call, e.g. getSecret("worker-password"): a lowercase name with no
# digits, directly inside the quotes of the first call argument. Real secrets passed
# positionally are rarely bare lowercase words, but the rule needs both conditions.
_NAME = re.compile(r'[a-z][a-z_\-./]{1,31}')
_CALL_OPEN = re.compile(r'[A-Za-z_][\w.]*\(\s*["\']$')


def lookup_argument(text, start, end):
    return bool(_NAME.fullmatch(text[start:end]) and _CALL_OPEN.search(text[max(0, start - 64):start])
                and text[end:end + 1] in ('"', "'"))


def filter_spans(text, spans):
    """Return (kept, dropped_rule_counts)."""
    kept, dropped = [], {}
    for s in spans:
        rule = non_value_rule(text[s['start']:s['end']])
        if rule is None and lookup_argument(text, s['start'], s['end']):
            rule = 'lookup_argument'
        if rule is None:
            kept.append(s)
        else:
            dropped[rule] = dropped.get(rule, 0) + 1
    return kept, dropped
