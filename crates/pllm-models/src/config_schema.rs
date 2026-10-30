// Private item emitter. The schema set lives at its parent invocation site;
// this engine has no model-family branches or default value implementations.

/// Private-engine fixtures use the same source include in positive and negative
/// doctests. Inclusion grants no public macro API or execution qualification.
///
/// Known named default (positive engine-access counterpart):
/// ```
/// # use serde::{Deserialize, Serialize};
/// # include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config_schema.rs"));
/// fn label() -> String { String::new() }
/// config_schema! { $; emit EMIT; derives config; struct Fixture; field_ids Field;
///     defaults { label => "label": Text = existing; }
///     fields { value: named(label); } }
/// EMIT!();
/// let _: Fixture = serde_json::from_str("{}").unwrap();
/// ```
/// Unknown named default:
/// ```compile_fail
/// # use serde::{Deserialize, Serialize};
/// # include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config_schema.rs"));
/// config_schema! { $; emit EMIT; derives config; struct Fixture; field_ids Field;
///     defaults {} fields { value: named(missing); } }
/// EMIT!();
/// ```
/// One default declaration (positive counterpart):
/// ```
/// # use serde::{Deserialize, Serialize};
/// # include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config_schema.rs"));
/// fn label() -> String { String::new() }
/// config_schema! { $; emit EMIT; derives config; struct Fixture; field_ids Field;
///     defaults { label => "label": Text = existing; }
///     fields { value: named(label); } }
/// EMIT!();
/// ```
/// Duplicate default name:
/// ```compile_fail
/// # use serde::{Deserialize, Serialize};
/// # include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config_schema.rs"));
/// fn label() -> String { String::new() }
/// config_schema! { $; emit EMIT; derives config; struct Fixture; field_ids Field;
///     defaults { label => "label": Text = existing; label => "label": Text = existing; }
///     fields { value: named(label); } }
/// EMIT!();
/// ```
/// Separate parent emitter names (positive counterpart):
/// ```
/// # use serde::{Deserialize, Serialize};
/// # include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config_schema.rs"));
/// config_schema! { $;
///     emit FIRST; derives config; struct First; field_ids FirstField;
///     defaults {} fields { value: required(U64); }
///     emit SECOND; derives config; struct Second; field_ids SecondField;
///     defaults {} fields { value: required(U64); } }
/// FIRST!(); SECOND!();
/// ```
/// Duplicate parent emitter name:
/// ```compile_fail
/// # use serde::{Deserialize, Serialize};
/// # include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config_schema.rs"));
/// config_schema! { $;
///     emit EMIT; derives config; struct First; field_ids FirstField;
///     defaults {} fields { value: required(U64); }
///     emit EMIT; derives config; struct Second; field_ids SecondField;
///     defaults {} fields { value: required(U64); } }
/// EMIT!();
/// ```
/// Matching identifier/path (positive counterpart):
/// ```
/// # use serde::{Deserialize, Serialize};
/// # include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config_schema.rs"));
/// fn label() -> String { String::new() }
/// fn other() -> String { String::new() }
/// config_schema! { $; emit EMIT; derives config; struct Fixture; field_ids Field;
///     defaults { label => "label": Text = existing; }
///     fields { value: named(label); } }
/// EMIT!();
/// ```
/// Mismatched path, with BOTH functions available (not a missing-function test):
/// ```compile_fail
/// # use serde::{Deserialize, Serialize};
/// # include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config_schema.rs"));
/// fn label() -> String { String::new() }
/// fn other() -> String { String::new() }
/// config_schema! { $; emit EMIT; derives config; struct Fixture; field_ids Field;
///     defaults { label => "other": Text = existing; }
///     fields { value: named(label); } }
/// EMIT!();
/// ```
/// Supported optional policy (positive counterpart):
/// ```
/// # use serde::{Deserialize, Serialize};
/// # include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config_schema.rs"));
/// config_schema! { $; emit EMIT; derives config; struct Fixture; field_ids Field;
///     defaults {} fields { value: optional(U64, implicit, keep); } }
/// EMIT!();
/// ```
/// Unsupported optional kind:
/// ```compile_fail
/// # use serde::{Deserialize, Serialize};
/// # include!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config_schema.rs"));
/// config_schema! { $; emit EMIT; derives config; struct Fixture; field_ids Field;
///     defaults {} fields { value: optional(Text, implicit, keep); } }
/// EMIT!();
/// ```
/// Existing public field access and types (external-crate visibility):
/// ```
/// fn fields(c: &pllm_models::QwenConfig) {
///     let _: &String = &c.model_type;
///     let _: &u64 = &c.hidden_size;
///     let _: &u64 = &c.intermediate_size;
///     let _: &u32 = &c.num_hidden_layers;
///     let _: &u32 = &c.num_attention_heads;
///     let _: &u32 = &c.num_key_value_heads;
///     let _: &u64 = &c.vocab_size;
///     let _: &u64 = &c.max_position_embeddings;
///     let _: &String = &c.hidden_act;
///     let _: &String = &c.rms_norm_eps;
///     let _: &u64 = &c.rope_theta;
///     let _: &bool = &c.tie_word_embeddings;
///     let _: &Option<u64> = &c.sliding_window;
///     let _: &Option<bool> = &c.use_sliding_window;
/// }
/// ```
macro_rules! config_schema {
    ($d:tt; $($entries:tt)*) => {
        config_schema! { @next $d; $($entries)* }
    };
    (@next $d:tt;) => {};
    (@next $d:tt;
        emit $e:ident; derives $derive:ident; struct $v:vis $s:ident;
        field_ids $f:ident;
        defaults { $($default:ident => $path:literal : $kind:ident = existing;)* }
        fields { $($fields:tt)* }
        $($entries:tt)*
    ) => {
        // Parent-scope marker: macro shadowing cannot hide duplicate emitters.
        const $e: () = ();
        const _: () = $e;
        $(config_schema! { @default_kind $kind })*
        const _: () = {
            $(
                #[allow(non_upper_case_globals)]
                const $default: () = ();
                let () = $default;
                {
                    let name = stringify!($default).as_bytes();
                    let path = $path.as_bytes();
                    assert!(name.len() == path.len());
                    let mut i = 0;
                    while i < name.len() {
                        assert!(name[i] == path[i]);
                        i += 1;
                    }
                }
            )*
        };
        // This lookup is a parent item, not trapped in the check-only block.
        // It is replaced by the resolved zero-argument emitter on completion.
        macro_rules! $e {
            $(
                (@lookup $default; $d($d rest:tt)*) => {
                    config_schema! { @named $kind $default $path existing $d($d rest)* }
                };
            )*
            (@lookup $d unknown:ident; $d($d rest:tt)*) => {
                compile_error!("unknown named config default");
            };
        }
        // One recursive walker hop per ordinary field avoids increasing the
        // crate's recursion limit. These are policy data, not model branches.
        config_schema! { @walker $d $e [$derive $v $s $f] [$($entries)*] [$($fields)*] {
            required(Text) => [::std::string::String] [];
            required(U64) => [::core::primitive::u64] [];
            required(U32) => [::core::primitive::u32] [];
            required(Bool) => [::core::primitive::bool] [];
            required(Decimal) => [::std::string::String] [#[serde(deserialize_with = "decimal_string")]];
            required(Integral) => [::core::primitive::u64] [#[serde(deserialize_with = "integral_u64")]];
            default(Bool) => [::core::primitive::bool] [#[serde(default)]];
            default(F64) => [::core::primitive::f64] [#[serde(default)]];
            optional(U32, implicit, keep) => [::std::option::Option<::core::primitive::u32>] [];
            optional(U32, implicit, skip_none) => [::std::option::Option<::core::primitive::u32>] [#[serde(skip_serializing_if = "Option::is_none")]];
            optional(U32, explicit, keep) => [::std::option::Option<::core::primitive::u32>] [#[serde(default)]];
            optional(U32, explicit, skip_none) => [::std::option::Option<::core::primitive::u32>] [#[serde(default, skip_serializing_if = "Option::is_none")]];
            optional(U64, implicit, keep) => [::std::option::Option<::core::primitive::u64>] [];
            optional(U64, implicit, skip_none) => [::std::option::Option<::core::primitive::u64>] [#[serde(skip_serializing_if = "Option::is_none")]];
            optional(U64, explicit, keep) => [::std::option::Option<::core::primitive::u64>] [#[serde(default)]];
            optional(U64, explicit, skip_none) => [::std::option::Option<::core::primitive::u64>] [#[serde(default, skip_serializing_if = "Option::is_none")]];
            optional(Bool, implicit, keep) => [::std::option::Option<::core::primitive::bool>] [];
            optional(Bool, implicit, skip_none) => [::std::option::Option<::core::primitive::bool>] [#[serde(skip_serializing_if = "Option::is_none")]];
            optional(Bool, explicit, keep) => [::std::option::Option<::core::primitive::bool>] [#[serde(default)]];
            optional(Bool, explicit, skip_none) => [::std::option::Option<::core::primitive::bool>] [#[serde(default, skip_serializing_if = "Option::is_none")]];
            optional(Json, implicit, keep) => [::std::option::Option<::serde_json::Value>] [];
            optional(Json, implicit, skip_none) => [::std::option::Option<::serde_json::Value>] [#[serde(skip_serializing_if = "Option::is_none")]];
            optional(Json, explicit, keep) => [::std::option::Option<::serde_json::Value>] [#[serde(default)]];
            optional(Json, explicit, skip_none) => [::std::option::Option<::serde_json::Value>] [#[serde(default, skip_serializing_if = "Option::is_none")]];
            optional(StringList, implicit, keep) => [::std::option::Option<::std::vec::Vec<::std::string::String>>] [];
            optional(StringList, implicit, skip_none) => [::std::option::Option<::std::vec::Vec<::std::string::String>>] [#[serde(skip_serializing_if = "Option::is_none")]];
            optional(StringList, explicit, keep) => [::std::option::Option<::std::vec::Vec<::std::string::String>>] [#[serde(default)]];
            optional(StringList, explicit, skip_none) => [::std::option::Option<::std::vec::Vec<::std::string::String>>] [#[serde(default, skip_serializing_if = "Option::is_none")]];
        } }
    };
    (@walker $d:tt $e:ident [$derive:ident $v:vis $s:ident $f:ident]
        [$($entries:tt)*] [$($fields:tt)*]
        { $($policy:ident ($($args:tt)*) => [$ty:ty] [$($attrs:tt)*];)* }
    ) => {
        macro_rules! $f {
            ($d dollar:tt; [$d($d out:tt)*] [$d($d ids:ident)*] []) => {
                config_schema! { @done $d dollar $e [$derive $v $s $f]
                    [$($entries)*] [$d($d out)*] [$d($d ids)*] }
            };
            $(
                ($d dollar:tt; [$d($d out:tt)*] [$d($d ids:ident)*] [
                    $d fv:vis $d name:ident : $policy($($args)*); $d($d todo:tt)*
                ]) => {
                    $f! { $d dollar;
                        [$d($d out)* $($attrs)* $d fv $d name: $ty,]
                        [$d($d ids)* $d name] [$d($d todo)*] }
                };
            )*
            ($d dollar:tt; [$d($d out:tt)*] [$d($d ids:ident)*] [
                $d fv:vis $d name:ident : named($d key:ident); $d($d todo:tt)*
            ]) => {
                $e! { @lookup $d key; [$d dollar $f
                    [$d($d out)*] [$d($d ids)*] [$d($d todo)*] [$d fv $d name]] }
            };
            ($d($d rest:tt)*) => {
                compile_error!("unsupported config field policy");
            };
        }
        $f! { $d; [] [] [$($fields)*] }
    };
    (@default_kind Text) => {};
    (@default_kind Decimal) => {};
    (@default_kind Integral) => {};
    (@named Text $default:ident $path:literal existing $cx:tt) => {
        config_schema! { @field $cx [::std::string::String] [#[serde(default = $path)]] }
    };
    (@named Decimal $default:ident $path:literal existing $cx:tt) => {
        config_schema! { @field $cx [::std::string::String]
            [#[serde(default = $path, deserialize_with = "decimal_string")]] }
    };
    (@named Integral $default:ident $path:literal existing $cx:tt) => {
        config_schema! { @field $cx [::core::primitive::u64]
            [#[serde(default = $path, deserialize_with = "integral_u64")]] }
    };
    (@field [$d:tt $f:ident [$($out:tt)*] [$($ids:ident)*]
        $todo:tt [$v:vis $name:ident]] [$ty:ty] [$($attrs:tt)*]
    ) => {
        $f! { $d; [$($out)* $($attrs)* $v $name: $ty,] [$($ids)* $name] $todo }
    };
    (@done $d:tt $e:ident [eq_config $v:vis $s:ident $f:ident] $entries:tt $out:tt $ids:tt) => {
        config_schema! { @emit $d $e [$v $s $f] $entries $out $ids
            [Clone, Debug, Deserialize, Eq, PartialEq, Serialize] }
    };
    (@done $d:tt $e:ident [config $v:vis $s:ident $f:ident] $entries:tt $out:tt $ids:tt) => {
        config_schema! { @emit $d $e [$v $s $f] $entries $out $ids
            [Clone, Debug, Deserialize, Serialize] }
    };
    (@emit $d:tt $e:ident [$v:vis $s:ident $f:ident] [$($entries:tt)*]
        [$($out:tt)*] [$($ids:ident)*] [$($derive:ident),*]
    ) => {
        macro_rules! $e {
            () => {
                #[derive($($derive),*)]
                $v struct $s { $($out)* }
                #[derive(Clone, Copy, Debug, Eq, PartialEq)]
                #[allow(non_camel_case_types)]
                enum $f { $($ids,)* }
                impl $f {
                    fn as_str(self) -> &'static ::core::primitive::str {
                        match self { $(Self::$ids => stringify!($ids),)* }
                    }
                }
            };
        }
        config_schema! { @next $d; $($entries)* }
    };
    ($($unsupported:tt)*) => {
        compile_error!("unsupported config schema declaration or policy");
    };
}
