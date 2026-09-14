//! In-memory state contract only. Durable multi-process allocation and rollback defenses
//! require the integration described in docs/SECURITY_ASSURANCE.md.
use std::fmt;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State { Ready, Bound, Exposed, Accepted, Retired }
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error { ChangedInput, InvalidTransition, Unbound }
impl fmt::Display for Error { fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { write!(f,"{self:?}") } }
impl std::error::Error for Error {}
/// Not Clone: cloning a public experiment is not cloning one-time material.
pub struct Record { id: u128, state: State, frozen: Option<Vec<u8>> }
impl fmt::Debug for Record {
    fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result { f.debug_struct("Record").field("id",&self.id).field("state",&self.state).field("payload",&"[redacted]").finish() }
}
impl Record {
    pub fn new(id:u128)->Self {Self{id,state:State::Ready,frozen:None}}
    pub fn state(&self)->State {self.state}
    pub fn bind(&mut self, payload:&[u8])->Result<(),Error> {
        match self.state {
            State::Ready=>{self.frozen=Some(payload.to_vec());self.state=State::Bound;Ok(())},
            State::Bound|State::Exposed=>{if self.frozen.as_deref()==Some(payload){Ok(())}else{Err(Error::ChangedInput)}},
            _=>Err(Error::InvalidTransition)
        }
    }
    /// Returns the already frozen application bytes. Transport creates its own fresh records.
    pub fn expose(&mut self)->Result<&[u8],Error> {
        if self.state!=State::Bound && self.state!=State::Exposed {return Err(Error::InvalidTransition)}
        self.state=State::Exposed; self.frozen.as_deref().ok_or(Error::Unbound)
    }
    pub fn accept(&mut self)->Result<(),Error> {if self.state!=State::Exposed{return Err(Error::InvalidTransition)}self.state=State::Accepted;Ok(())}
    pub fn retire(&mut self) {self.state=State::Retired;self.frozen=None;}
}
#[cfg(test)]mod tests {
    use super::*;
    #[test] fn retry_identical() {let mut r=Record::new(1);r.bind(b"same").unwrap();assert_eq!(r.expose().unwrap(),b"same");r.bind(b"same").unwrap();assert_eq!(r.expose().unwrap(),b"same");}
    #[test] fn changing_input_rejected() {let mut r=Record::new(1);r.bind(b"a").unwrap();r.expose().unwrap();assert_eq!(r.bind(b"b"),Err(Error::ChangedInput));}
    #[test] fn retired_cannot_rebind() {let mut r=Record::new(1);r.retire();assert!(r.bind(b"a").is_err());}
    #[test] fn no_double_accept() {let mut r=Record::new(1);r.bind(b"a").unwrap();r.expose().unwrap();r.accept().unwrap();assert!(r.accept().is_err());}
    #[test] fn no_expose_before_bind() {assert!(Record::new(2).expose().is_err());}
    #[test] fn debug_redacts() {let mut r=Record::new(1);r.bind(b"secret fixture").unwrap();assert!(!format!("{r:?}").contains("secret fixture"));}
}
